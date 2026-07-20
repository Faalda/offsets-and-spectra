"""
Reviewer request: does REGULARIZING the linear map close the gap to Procrustes/CCA
under ~1k pairs? We add ridge-regularized linear regression (lambda sweep) and PLS
to the alignment ladder, across the 12 cross-lingual pairs, vs unregularized
linear-ls, the orthogonal map, and CCA.

Tests the bias-variance narrative directly: if regularized-linear improves over
linear-ls but still trails ortho/CCA, structure (not just regularization) is what
matters.

Run: nadi/.venv/bin/python3 experiments/regularized_baselines.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual            # noqa: E402

SRC = "intfloat/multilingual-e5-base"
PAIRS = [("deu_Latn", "bert-base-german-cased", "de"), ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "ru"),
         ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "ar"), ("zho_Hans", "bert-base-chinese", "zh"),
         ("fra_Latn", "camembert-base", "fr"), ("spa_Latn", "dccuchile/bert-base-spanish-wwm-cased", "es"),
         ("ita_Latn", "dbmdz/bert-base-italian-cased", "it"), ("por_Latn", "neuralmind/bert-base-portuguese-cased", "pt"),
         ("nld_Latn", "GroNLP/bert-base-dutch-cased", "nl"), ("tur_Latn", "dbmdz/bert-base-turkish-cased", "tr"),
         ("fin_Latn", "TurkuNLP/bert-base-finnish-cased-v1", "fi"), ("pes_Arab", "HooshvareLab/bert-base-parsbert-uncased", "fa")]
LAMBDAS = [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PLS_COMPONENTS = [64, 128, 256]
PLS_WARNING_COUNT = 0


def p1(M, Y):
    a, b = F.normalize(M, dim=-1), F.normalize(Y, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


def ortho(a, b, c, e):
    Xc, Yc = a - a.mean(0), b - b.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return p1(c @ (U @ Vt), e)


def ridge(a, b, c, e, lam):
    mx, my = a.mean(0), b.mean(0); Xc, Yc = a - mx, b - my
    d = Xc.shape[1]
    W = torch.linalg.solve(Xc.T @ Xc + lam * Xc.shape[0] * torch.eye(d, device=Xc.device), Xc.T @ Yc)
    return p1((c - mx) @ W + my, e)


def pls(a, b, c, e, k):
    global PLS_WARNING_COUNT
    from sklearn.cross_decomposition import PLSRegression
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", ConvergenceWarning)
        m = PLSRegression(n_components=k, scale=False, max_iter=5000, tol=1e-5).fit(a.numpy(), b.numpy())
    PLS_WARNING_COUNT += sum(1 for w in ws if issubclass(w.category, ConvergenceWarning))
    return p1(torch.tensor(m.predict(c.numpy()), dtype=torch.float32), e)


def cca(a, b, c, e, lam=0.1):
    n = a.shape[0]; mx, my = a.mean(0), b.mean(0); Xc, Yc = a - mx, b - my
    d = Xc.shape[1]; I = torch.eye(d)
    Cxx = Xc.T@Xc/n+lam*I; Cyy = Yc.T@Yc/n+lam*I; Cxy = Xc.T@Yc/n
    def isq(C):
        w, V = torch.linalg.eigh(C); return V@torch.diag(w.clamp(min=1e-6)**-0.5)@V.T
    Mx, My = isq(Cxx), isq(Cyy); U, S, Vt = torch.linalg.svd(Mx@Cxy@My)
    return p1((c-mx)@(Mx@U), (e-my)@(My@Vt.T))


def split_train_val(a, b, val_frac: float = 0.2, seed: int = 0):
    n = a.shape[0]
    g = torch.Generator(device=a.device if a.is_cuda else "cpu")
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=a.device)
    n_val = max(64, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    val = perm[:n_val]
    tr = perm[n_val:]
    return a[tr], b[tr], a[val], b[val]


def main():
    print(f"{'pair':<5}{'ortho':>7}{'linls':>7}{'ridge*':>8}{'PLS*':>7}{'CCA':>7}")
    agg = {k: [] for k in ["ortho", "linls", "ridge", "pls", "cca"]}
    for tgt, enc, lab in PAIRS:
        d = load_crosslingual("eng_Latn", tgt, SRC, "cpu", encoder_tgt=enc)
        a, b, c, e = d.X_train, d.Y_train, d.X_test, d.Y_test
        atr, btr, aval, bval = split_train_val(a, b, val_frac=0.2, seed=0)
        o = ortho(a, b, c, e); ll = ridge(a, b, c, e, 0.0 + 1e-12)
        best_lam = max(LAMBDAS, key=lambda lm: ridge(atr, btr, aval, bval, lm))
        rg = ridge(a, b, c, e, best_lam)
        best_k = max(PLS_COMPONENTS, key=lambda k: pls(atr, btr, aval, bval, k))
        pl = pls(a, b, c, e, best_k)
        cc = cca(a, b, c, e)
        for k, v in zip(agg, [o, ll, rg, pl, cc]): agg[k].append(v)
        print(f"{lab:<5}{o:>7.3f}{ll:>7.3f}{rg:>8.3f}{pl:>7.3f}{cc:>7.3f}")
    print("-" * 41)
    print(f"{'avg':<5}" + "".join(f"{np.mean(agg[k]):>7.3f}" if k != "ridge" else f"{np.mean(agg[k]):>8.3f}"
                                   for k in ["ortho", "linls", "ridge", "pls", "cca"]))
    print("(* = best over lambda / n_components)")
    if PLS_WARNING_COUNT:
        print(f"(note: {PLS_WARNING_COUNT} internal PLS convergence warnings during selection; output was stabilized for clean logs)")


if __name__ == "__main__":
    main()
