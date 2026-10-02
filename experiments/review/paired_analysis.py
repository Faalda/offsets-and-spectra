"""Paired bootstrap analysis for the AISTATS alignment manuscript.

Reports query-paired CCA-versus-centred-Procrustes P@1 differences across the
41 text development gaps. It also evaluates the identity branch on a disjoint
CLIP/MS-COCO sample split excluded from the original CLIP development instance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "review"))

import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402

N_BOOT = 2_000
SEED = 0


def paired_ci(delta: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(delta)
    means = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return [float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def correctness(m, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    fx, fy = m
    return C.p1_vec(fx(X), fy(Y)).cpu().numpy().astype(np.float64)


def text_bootstrap(rng: np.random.Generator) -> dict:
    rows = {}
    for group in ("xling_e5", "xling_labse", "xarch"):
        for name, _, Xtr, Ytr, Xte, Yte in gaps(group):
            Xtr, Ytr, Xte, Yte = (x.float() for x in (Xtr, Ytr, Xte, Yte))
            proc = H.centred_map(C.ortho_map(Xtr, Ytr), Xtr, Ytr)
            cca, lam, _ = H.cca_val_select(Xtr, Ytr)
            delta = correctness(cca, Xte, Yte) - correctness(proc, Xte, Yte)
            rows[name] = {
                "n_test": int(len(delta)),
                "lambda": lam,
                "cca_minus_procrustes": paired_ci(delta, rng),
            }
    # Report a gap-level bootstrap, which matches the paper's unit of analysis.
    gap_means = np.array([row["cca_minus_procrustes"][0] for row in rows.values()])
    means = np.array([gap_means[rng.integers(0, len(gap_means), len(gap_means))].mean()
                      for _ in range(N_BOOT)])
    return {
        "n_gaps": len(rows),
        "gap_mean_cca_minus_procrustes": [
            float(gap_means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        ],
        "per_gap": rows,
    }


def clip_identity_holdout(rng: np.random.Generator) -> dict:
    cache = ROOT / "results" / "emb_cache" / "clip_mscoco.pt"
    data = torch.load(cache, weights_only=False)
    X, Y = data["img"].float(), data["txt"].float()
    # The main manuscript uses [0:1000] to fit and [1000:1500] to evaluate
    # this instance. Extend the public stream to obtain a separate sample split.
    if len(X) < 2_000:
        from experiments.crossmodal_clip import embed_clip
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        device = "cuda:0"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        # Remove the old short cache so embed_clip can replace it with 2,000 rows.
        cache.unlink()
        X, Y = embed_clip("mscoco", "clip-benchmark/wds_mscoco_captions", "test",
                           n_train=1_500, n_test=1_000, processor=processor, model=model)
        del model
        torch.cuda.empty_cache()
    if len(X) < 2_000:
        return {
            "status": "not_run",
            "reason": "Could not obtain 2,000 public CLIP/MS-COCO pairs.",
            "cached_pairs": int(len(X)),
            "required_pairs": 2_000,
        }
    Xtr, Ytr, Xte, Yte = X[1500:2000], Y[1500:2000], X[2000:2500], Y[2000:2500]
    if len(Xte) == 0:
        # A 2,000-pair cache still permits a disjoint fit/evaluation instance.
        Xtr, Ytr, Xte, Yte = X[1500:1750], Y[1500:1750], X[1750:2000], Y[1750:2000]
    identity = correctness(C.identity_map(Xtr, Ytr), Xte, Yte)
    cca, lam, _ = H.cca_val_select(Xtr, Ytr)
    cca_correct = correctness(cca, Xte, Yte)
    decision = "identity" if C.p1(Xtr, Ytr) >= 0.5 else "cca"
    chosen = identity if decision == "identity" else cca_correct
    return {
            "scope": "Disjoint CLIP/MS-COCO sample split; not a new encoder pair or dataset.",
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "identity_train_p1": C.p1(Xtr, Ytr),
        "identity_test_p1_ci95": paired_ci(identity, rng),
        "cca_test_p1_ci95": paired_ci(cca_correct, rng),
        "identity_minus_cca_ci95": paired_ci(identity - cca_correct, rng),
        "selected": decision,
        "selected_regret": float(max(identity.mean(), cca_correct.mean()) - chosen.mean()),
        "cca_lambda": lam,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {
        "bootstrap_resamples": N_BOOT,
        "seed": SEED,
        "text_cca_vs_procrustes": text_bootstrap(rng),
        "clip_identity_holdout": clip_identity_holdout(rng),
    }
    path = ROOT / "results" / "review" / "paired_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
