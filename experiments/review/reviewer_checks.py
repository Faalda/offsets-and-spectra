"""Targeted checks requested in paper/notes.txt, using cached embeddings only.

Usage:
  python experiments/review/reviewer_checks.py summary
  python experiments/review/reviewer_checks.py modernbert
  python experiments/review/reviewer_checks.py GROUP

GROUP is any group accepted by benchmark.gaps. Results are written under
results/review/reviewer_checks_*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "review"))

import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402

RESULTS = ROOT / "results" / "review"
Laya_CACHE = ROOT / "results" / "emb_cache" / "laya"
EMBED_CACHE = ROOT / "results" / "emb_cache"


def raw_fit_map(X, Y):
    U, _, Vt = torch.linalg.svd(X.T @ Y, full_matrices=False)
    R = U @ Vt
    return (lambda x: x @ R), (lambda y: y)


def scaled_map(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0).clamp_min(1e-8)
    sy = Y.std(0).clamp_min(1e-8)
    Xs, Ys = (X - mx) / sx, (Y - my) / sy
    U, _, Vt = torch.linalg.svd(Xs.T @ Ys, full_matrices=False)
    R = U @ Vt
    return (lambda x: ((x - mx) / sx) @ R), (lambda y: (y - my) / sy)


def mean_dominance(Z):
    return float(Z.mean(0).square().sum() / Z.square().sum(1).mean())


def offset_direction_scale(X, Y, rotation):
    fx, _ = rotation
    offset = fx(X.mean(0, keepdim=True)).squeeze(0)
    direction = offset / offset.norm().clamp_min(1e-12)
    Yc = Y - Y.mean(0)
    directional_sd = (Yc @ direction).square().mean().sqrt()
    return {
        "mean_dominance_x": mean_dominance(X),
        "relative_offset_norm": float(
            X.mean(0).norm() / (X - X.mean(0)).square().sum(1).mean().sqrt()
        ),
        "target_directional_sd": float(directional_sd),
        "direction_aware_offset": float(offset.norm() * directional_sd),
    }


def evaluate(Xtr, Ytr, Xte, Yte):
    Xtr, Ytr, Xte, Yte = (z.float().cpu() for z in (Xtr, Ytr, Xte, Yte))
    rotation = C.ortho_map(Xtr, Ytr)
    centred = H.centred_map(rotation, Xtr, Ytr)
    cca, selected_lambda, validation_curve = H.cca_val_select(Xtr, Ytr)
    maps = {
        "identity_native": C.identity_map(Xtr, Ytr) if Xtr.shape[1] == Ytr.shape[1] else None,
        "probe": rotation,
        "raw_fit_procrustes": raw_fit_map(Xtr, Ytr),
        "centred_procrustes": centred,
        "scaled_procrustes": scaled_map(Xtr, Ytr),
        "cca_selected": cca,
        "cca_fixed_0.001": C.cca_map(Xtr, Ytr, 1e-3),
        "cca_large_10": C.cca_map(Xtr, Ytr, 10.0),
    }
    p1 = {name: C.evalmap(mapping, Xte, Yte) for name, mapping in maps.items() if mapping is not None}
    p1["probe_csls"] = H.csls_p1(rotation, Xte, Yte)
    return {
        "shape": {"train": list(Xtr.shape), "target_train": list(Ytr.shape), "test": list(Xte.shape)},
        "p1": p1,
        "selected_lambda": selected_lambda,
        "validation_curve": {str(k): v for k, v in validation_curve.items()},
        "offset": offset_direction_scale(Xtr, Ytr, rotation),
    }


def evaluate_classical(Xtr, Ytr, Xte, Yte):
    Xtr, Ytr, Xte, Yte = (z.float().cpu() for z in (Xtr, Ytr, Xte, Yte))
    rotation = C.ortho_map(Xtr, Ytr)
    maps = {
        "identity_native": C.identity_map(Xtr, Ytr) if Xtr.shape[1] == Ytr.shape[1] else None,
        "probe": rotation,
        "raw_fit_procrustes": raw_fit_map(Xtr, Ytr),
        "centred_procrustes": H.centred_map(rotation, Xtr, Ytr),
        "scaled_procrustes": scaled_map(Xtr, Ytr),
    }
    p1 = {name: C.evalmap(mapping, Xte, Yte) for name, mapping in maps.items() if mapping is not None}
    p1["probe_csls"] = H.csls_p1(rotation, Xte, Yte)
    return {
        "shape": {"train": list(Xtr.shape), "target_train": list(Ytr.shape), "test": list(Xte.shape)},
        "p1": p1,
        "offset": offset_direction_scale(Xtr, Ytr, rotation),
    }


def modernbert():
    Xtr = torch.load(Laya_CACHE / "flores_modernbert-base-large_dev.pt")
    Ytr = torch.load(Laya_CACHE / "flores_laya-modernbert_dev.pt")
    Xte = torch.load(Laya_CACHE / "flores_modernbert-base-large_devtest.pt")
    Yte = torch.load(Laya_CACHE / "flores_laya-modernbert_devtest.pt")
    return {"laya/modernbert-base->laya-modernbert": evaluate(Xtr, Ytr, Xte, Yte)}


def corrected_slurp():
    whisper = "openai_whisper-large-v3"
    wav2vec = "jonatasgrosman_wav2vec2-large-xlsr-53-arabic"
    text = "xlm-roberta-large"

    def load(kind, model, split):
        return torch.load(EMBED_CACHE / f"slurptn__{kind}__{model}__{split}.pt")

    Xtr = torch.cat([
        load("speech-validframes-fp32", whisper, "train"),
        load("speech-validframes-fp32", whisper, "validation"),
    ])
    Ytr = torch.cat([load("text", text, "train"), load("text", text, "validation")])
    Xte = load("speech-validframes-fp32", whisper, "test")
    Yte = load("text", text, "test")
    Vtr = torch.cat([load("speech", wav2vec, "train"), load("speech", wav2vec, "validation")])
    Vte = load("speech", wav2vec, "test")
    _, inverse = torch.unique(Yte, dim=0, return_inverse=True)
    first = {}
    for index, group in enumerate(inverse.tolist()):
        first.setdefault(group, index)
    keep = torch.tensor(sorted(first.values()))
    return {
        "slurp/whisper-validframes->wav2vec2": evaluate(Xtr, Vtr, Xte, Vte),
        "slurp/whisper-validframes->xlmr": evaluate(Xtr, Ytr, Xte[keep], Yte[keep]),
    }


def corrected_librispeech():
    speech = torch.load(
        EMBED_CACHE / "librispeech__speech-validframes-fp32__openai_whisper-large-v3__test.pt"
    )
    text = torch.load(EMBED_CACHE / "librispeech__text__xlm-roberta-large__test.pt")
    output = {}
    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(speech), generator=generator)
        n_train = int(0.8 * len(speech))
        train, test = permutation[:n_train], permutation[n_train:]
        output[f"librispeech/whisper-validframes->xlmr/s{seed}"] = evaluate(
            speech[train], text[train], speech[test], text[test]
        )
    return output


def aggregate_existing():
    hub = json.loads((RESULTS / "hub_dev.json").read_text())
    decomposition = json.loads((RESULTS / "decomposition.json").read_text())
    rows = [row for row in decomposition["rows"] if row["regime"] in ("cross-lingual", "cross-arch")]
    records = []
    for row in rows:
        result = hub[row["gap"]]
        p1 = result["p1"]
        offset_cost = p1["rotation+centre"] - p1["rotation"]
        records.append({
            "gap": row["gap"],
            "A": row["A"],
            "whitening_gain": p1["cca-val"] - p1["rotation+centre"],
            "selected_lambda": result["lam"]["val"],
            "fixed_0.001_gain": result["test_curve"]["0.001"] - p1["rotation+centre"],
            "large_10_gap_from_centred": result["test_curve"]["10.0"] - p1["rotation+centre"],
            "offset_cost": offset_cost,
            "probe_csls_recovery": p1["rotation+csls"] - p1["rotation"],
            "probe_csls_fraction_of_centring": (
                (p1["rotation+csls"] - p1["rotation"]) / offset_cost
                if abs(offset_cost) > 1e-12 else None
            ),
        })
    gain = np.array([row["whitening_gain"] for row in records])
    selected = np.log10([row["selected_lambda"] for row in records])
    mismatch = np.array([row["A"] for row in records])
    lambda_rho, lambda_p = spearmanr(selected, gain)
    mismatch_rho, mismatch_p = spearmanr(mismatch, gain)
    fractions = [row["probe_csls_fraction_of_centring"] for row in records
                 if row["probe_csls_fraction_of_centring"] is not None]
    output = {
        "scope": "41 text gaps",
        "selected_lambda_vs_whitening_gain": {"rho": float(lambda_rho), "p": float(lambda_p)},
        "spectral_mismatch_vs_whitening_gain": {"rho": float(mismatch_rho), "p": float(mismatch_p)},
        "fixed_0.001_gain_mean": float(np.mean([row["fixed_0.001_gain"] for row in records])),
        "large_10_gap_from_centred_mean": float(np.mean([row["large_10_gap_from_centred"] for row in records])),
        "probe_csls_fraction_of_centring_median": float(np.median(fractions)),
        "records": records,
    }
    classical = []
    for group in ("xling_e5", "xling_labse", "xarch"):
        path = RESULTS / f"reviewer_checks_{group}.json"
        if path.exists():
            classical.extend(json.loads(path.read_text()).values())
    if classical:
        offset_cost = np.array([row["p1"]["centred_procrustes"] - row["p1"]["probe"] for row in classical])
        mean_dominance = np.array([row["offset"]["mean_dominance_x"] for row in classical])
        directional = np.array([row["offset"]["direction_aware_offset"] for row in classical])
        md_rho, md_p = spearmanr(mean_dominance, offset_cost)
        directional_rho, directional_p = spearmanr(directional, offset_cost)
        method_means = {
            method: float(np.mean([row["p1"][method] for row in classical]))
            for method in ("probe", "raw_fit_procrustes", "centred_procrustes", "scaled_procrustes", "probe_csls")
        }
        output["classical_checks"] = {
            "n": len(classical),
            "mean_p1": method_means,
            "raw_fit_beats_probe_count": int(sum(
                row["p1"]["raw_fit_procrustes"] > row["p1"]["probe"] for row in classical
            )),
            "raw_fit_beats_centred_count": int(sum(
                row["p1"]["raw_fit_procrustes"] > row["p1"]["centred_procrustes"] for row in classical
            )),
            "mean_dominance_vs_offset_cost": {"rho": float(md_rho), "p": float(md_p)},
            "direction_aware_offset_vs_offset_cost": {"rho": float(directional_rho), "p": float(directional_p)},
        }
    return output


def main():
    mode = sys.argv[1]
    if mode == "summary":
        output = aggregate_existing()
    elif mode == "modernbert":
        output = modernbert()
    elif mode == "slurp_corrected":
        output = corrected_slurp()
    elif mode == "librispeech_corrected":
        output = corrected_librispeech()
    else:
        output = {}
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(mode):
            output[name] = evaluate_classical(Xtr, Ytr, Xte, Yte)
            output[name]["regime"] = regime
            print(name, {k: round(v, 4) for k, v in output[name]["p1"].items()}, flush=True)
    path = RESULTS / f"reviewer_checks_{mode}.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output if mode == "summary" else {"saved": str(path), "n": len(output)}, indent=2))


if __name__ == "__main__":
    main()