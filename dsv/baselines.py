"""
Closed-form linear alignment baselines.

These give analytic reference points for the cross-lingual alignment task:

    procrustes(X, Y)  — optimal ORTHOGONAL map (rotation + centering). This is the
                        exact optimum of OrthoDSV's hypothesis class, so a correctly
                        trained `ortho` controller should match it.
    linear_ls(X, Y)   — optimal UNCONSTRAINED linear map (least squares + bias). The
                        optimum of the `affine` / `matrix` class; a linear upper bound.

Both return a callable `f` mapping source embeddings to the target space.
"""
from __future__ import annotations

import torch


def procrustes(X: torch.Tensor, Y: torch.Tensor):
    """Optimal orthogonal map with centering: argmin_{RᵀR=I} ‖(X-μx)R - (Y-μy)‖.

    Solution R = U Vᵀ from SVD of (X-μx)ᵀ(Y-μy). Returns f(x) = (x-μx)R + μy."""
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_x, Y - mu_y
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return lambda x: (x - mu_x) @ R + mu_y


def linear_ls(X: torch.Tensor, Y: torch.Tensor):
    """Optimal unconstrained affine map: argmin_{W,b} ‖XW + b - Y‖ (least squares)."""
    ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
    Xa = torch.cat([X, ones], dim=1)            # augment with bias column
    W = torch.linalg.lstsq(Xa, Y).solution      # (d+1, d)
    return lambda x: torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1) @ W
