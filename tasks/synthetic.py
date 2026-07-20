"""
Stage-1 task: controllable domain shift on synthetic Gaussian mixtures.

We control the ground-truth transform between a *source* domain (on which a
classifier is trained and then frozen) and a *target* domain (a known
translation and/or rotation of the source manifold, plus noise). A DSV
controller is then trained — with the backbone/classifier frozen — to map target
representations back so the source classifier works on them.

Because the true shift is known, this task answers precisely which
parametrisations can recover which kinds of shift:

    translation-only  → dsv already suffices; matrix/diag/scaled tie it (absorbed)
    rotation present   → dsv cannot fix it; ortho / affine / lowrank can
    both               → ortho should match the dense adapters at a fraction of params
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def random_rotation(dim: int, generator: torch.Generator) -> torch.Tensor:
    """A uniformly-random rotation matrix (QR of a Gaussian, det fixed to +1)."""
    a = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)   # make QR unique
    if torch.det(q) < 0:                                 # force SO(d)
        q[:, 0] = -q[:, 0]
    return q


@dataclass
class DomainShiftData:
    Xs_train: torch.Tensor
    ys_train: torch.Tensor
    Xt_train: torch.Tensor
    yt_train: torch.Tensor
    Xt_test: torch.Tensor
    yt_test: torch.Tensor
    R_true: torch.Tensor
    t_true: torch.Tensor
    dim: int
    n_classes: int


def make_domain_shift(
    n_classes: int = 6,
    dim: int = 32,
    n_per_class: int = 400,
    shift: str = "both",          # "translation" | "rotation" | "both" | "none"
    class_sep: float = 2.5,
    noise: float = 0.4,
    trans_scale: float = 4.0,
    seed: int = 0,
) -> DomainShiftData:
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, dim, generator=g) * class_sep

    def sample(n):
        xs, ys = [], []
        for c in range(n_classes):
            xs.append(means[c] + noise * torch.randn(n, dim, generator=g))
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    Xs_train, ys_train = sample(n_per_class)          # source (classifier trained here)
    Xt_raw, yt = sample(n_per_class)                  # target, pre-shift
    Xt_te_raw, yt_te = sample(n_per_class // 2)

    R_true = (random_rotation(dim, g) if shift in ("rotation", "both")
              else torch.eye(dim))
    t_true = (torch.randn(dim, generator=g) * trans_scale if shift in ("translation", "both")
              else torch.zeros(dim))

    # target = rotate then translate the source manifold
    Xt_train = Xt_raw @ R_true.T + t_true
    Xt_test = Xt_te_raw @ R_true.T + t_true

    return DomainShiftData(
        Xs_train, ys_train, Xt_train, yt, Xt_test, yt_te,
        R_true, t_true, dim, n_classes,
    )
