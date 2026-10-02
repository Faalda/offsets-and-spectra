"""Shared pieces for the nonlinear-gap stress tests (twist.py, two_centre.py).

Everything runs in the top-K principal components of one real source encoder
(multilingual-e5-base on FLORES English), so a constructed transformation acts
on ALL the variance and per-group fits are determined (each group has more
points than dimensions).  Data are centred, so the offset factor is absent by
design: these experiments isolate nonlinear correspondence distortion.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from source_families import ld  # noqa: E402

SRC = "intfloat_multilingual-e5-base"
K = 128


def load_base(k=K):
    """Return centred top-k PC coordinates: (Ztr, Zte).  PCA fitted on FLORES dev."""
    Xtr, Xte = ld(SRC, "eng_Latn", "dev"), ld(SRC, "eng_Latn", "devtest")
    mu = Xtr.mean(0)
    _, _, Vt = torch.linalg.svd(Xtr - mu, full_matrices=False)
    V = Vt[:k].T
    return (Xtr - mu) @ V, (Xte - mu) @ V


def mean_shift(X, Y):
    """||mu_Y - mu_X|| relative to the RMS norm of centred X."""
    rms = float((X - X.mean(0)).pow(2).sum(1).mean().sqrt())
    return float((Y.mean(0) - X.mean(0)).norm()) / rms


def cov_gap(X, Y):
    """Relative Frobenius distance between the full covariances (not just eigenvalues)."""
    Sx = (X - X.mean(0)).T @ (X - X.mean(0)) / X.shape[0]
    Sy = (Y - Y.mean(0)).T @ (Y - Y.mean(0)) / Y.shape[0]
    return float((Sx - Sy).norm() / Sx.norm())


# ------------------------------------------------------------------ assignments
def make_assign(kind, Xtr, G):
    """Label-free grouping computed from the SOURCE only, applicable to new points.
    kind='pc1'  : G quantile bins of the first principal coordinate.
    kind='norm' : G quantile bins of the centred norm."""
    f = (lambda x: x[:, 0]) if kind == "pc1" else (lambda x: x.norm(dim=1))
    edges = torch.quantile(f(Xtr), torch.linspace(0, 1, G + 1)[1:-1])
    return lambda x: torch.bucketize(f(x), edges)


# ------------------------------------------------------------------ maps
def rot_fit(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = torch.linalg.svd((X - mx).T @ (Y - my), full_matrices=False)
    return mx, my, U @ Vt


def global_procrustes(Xtr, Ytr):
    """Centred Procrustes with retrieval-side centring (the paper's baseline)."""
    return H.centred_map(C.ortho_map(Xtr, Ytr), Xtr, Ytr)


def pw_map(Xtr, Ytr, assign, G):
    """Piecewise centred Procrustes: one rotation and pair of means per group."""
    g_tr = assign(Xtr)
    params = {}
    for g in range(G):
        idx = (g_tr == g).nonzero().squeeze(1)
        if idx.numel() < 2:
            continue
        params[g] = rot_fit(Xtr[idx], Ytr[idx])
    fallback = rot_fit(Xtr, Ytr)

    def fx(x):
        gs = assign(x)
        out = torch.empty_like(x)
        for g in range(G):
            sel = (gs == g).nonzero().squeeze(1)
            if sel.numel() == 0:
                continue
            mx, my, R = params.get(g, fallback)
            out[sel] = (x[sel] - mx) @ R + my
        return out
    return fx, (lambda y: y)


def cv_gain(Xtr, Ytr, assign_kind, G, seed=0):
    """Label-free structure diagnostic D: extra variance explained, on a held-out 20%
    of the TRAINING pairs, by piecewise over global centred Procrustes,
    D = (MSE_global - MSE_piecewise) / E||y_c||^2.  Needs no test data."""
    a, b, va, vb = C.split_val(Xtr, Ytr, seed=seed)
    mx, my, R = rot_fit(a, b)
    mse_g = float((((va - mx) @ R + my) - vb).pow(2).sum(1).mean())
    fx, _ = pw_map(a, b, make_assign(assign_kind, a, G), G)
    mse_p = float((fx(va) - vb).pow(2).sum(1).mean())
    tot = float((vb - vb.mean(0)).pow(2).sum(1).mean())
    return (mse_g - mse_p) / tot


DIAG_FAMILIES = [("pc1", 2), ("pc1", 4), ("norm", 2), ("norm", 4)]


def diagnostic_D(Xtr, Ytr, seed=0):
    """Max over a fixed, generic family of label-free groupings."""
    return max(cv_gain(Xtr, Ytr, k, G, seed) for k, G in DIAG_FAMILIES)
