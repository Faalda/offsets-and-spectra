"""
Bootstrap 95% confidence intervals for cross-lingual alignment retrieval P@1.

The alignment maps we report (orthogonal Procrustes, least-squares linear) are
*deterministic* closed-form solutions, so training-seed error bars would be ~0
and meaningless. The honest uncertainty here is over the finite test set: we
resample the test queries with replacement and report the 95% CI of P@1. The
candidate set is held fixed (all N targets); only the queries averaged over are
resampled — a standard bootstrap of a retrieval-accuracy estimate.

Run:
    nadi/.venv/bin/python3 experiments/bootstrap_align.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv.baselines import linear_ls                       # noqa: E402
from tasks import load_crosslingual                       # noqa: E402


def ortho_map(Xtr, Ytr):
    """Our `ortho` controller realized in closed form: rotation R from the centered
    Procrustes SVD, applied without re-centering (matches the controller forward and
    is the better estimator for cosine retrieval)."""
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return lambda x: x @ R

PAIRS = [
    ("deu_Latn", "bert-base-german-cased", "en->de"),
    ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "en->ru"),
    ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en->ar"),
    ("zho_Hans", "bert-base-chinese", "en->zh"),
]
N_BOOT = 2000
SEED = 0


def per_query_correct(mapped, target) -> np.ndarray:
    a = F.normalize(mapped, dim=-1)
    b = F.normalize(target, dim=-1)
    pred = (a @ b.T).argmax(1)
    gold = torch.arange(a.shape[0])
    return (pred == gold).cpu().numpy().astype(np.float64)


def boot_ci(correct: np.ndarray, rng) -> tuple:
    n = len(correct)
    means = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return correct.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    rng = np.random.default_rng(SEED)
    print(f"\nBootstrap 95% CI of retrieval P@1 ({N_BOOT} resamples)")
    print("-" * 72)
    print(f"{'pair':<8}{'identity':>20}{'ortho':>22}{'linear-ls':>22}")
    print("-" * 72)
    for tgt, enc_t, label in PAIRS:
        try:
            d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base",
                                  "cpu", encoder_tgt=enc_t)
        except Exception as e:
            print(f"{label:<8}  (skipped: {type(e).__name__}: {str(e)[:40]})")
            continue
        Xtr, Ytr, Xte, Yte = d.X_train, d.Y_train, d.X_test, d.Y_test
        cells = {}
        cells["identity"] = boot_ci(per_query_correct(Xte, Yte), rng)
        cells["ortho"] = boot_ci(per_query_correct(ortho_map(Xtr, Ytr)(Xte), Yte), rng)
        cells["lin"] = boot_ci(per_query_correct(linear_ls(Xtr, Ytr)(Xte), Yte), rng)
        fmt = lambda c: f"{c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]"
        print(f"{label:<8}{fmt(cells['identity']):>20}{fmt(cells['ortho']):>22}{fmt(cells['lin']):>22}")
    print("-" * 72)


if __name__ == "__main__":
    main()
