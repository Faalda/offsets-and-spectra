"""
Hubness tools for the hub-aware extension.

hubness(Q, T, k)   skewness of the k-occurrence distribution N_k: how often each
                   target is among the k nearest neighbours of the queries
                   (Radovanovic et al. 2010).  Uses NO correspondence labels.
csls_p1            retrieval with CSLS (Conneau et al. 2018): 2cos - r_T(x) - r_S(y)
                   computed transductively over the evaluation pool.
cca_hub_select     choose CCA's ridge lambda by MINIMISING hubness of the mapped
                   queries against the targets.  Label-free given the fitted map:
                   'in' = on the training pairs themselves (all pairs used to fit,
                   no validation split); 'unpaired' = on held-out source and target
                   samples used only as two unpaired clouds.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import skew

import common as C

LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _cos(A, B):
    return F.normalize(A.float(), dim=-1) @ F.normalize(B.float(), dim=-1).T


def hubness(Q, T, k=10):
    S = _cos(Q, T)
    k = min(k, T.shape[0] - 1)
    idx = S.topk(k, dim=1).indices.flatten()
    n = torch.bincount(idx, minlength=T.shape[0]).double().numpy()
    return float(skew(n)) if n.std() > 0 else 0.0


def map_hubness(m, X, Y, k=10):
    fx, fy = m
    return hubness(fx(X), fy(Y), k)


def csls_p1(m, X, Y, k=10):
    fx, fy = m
    S = _cos(fx(X), fy(Y))
    k = min(k, S.shape[0] - 1)
    rT = S.topk(k, dim=1).values.mean(1, keepdim=True)
    rS = S.topk(k, dim=0).values.mean(0, keepdim=True)
    c = 2 * S - rT - rS
    return (c.argmax(1) == torch.arange(S.shape[0])).float().mean().item()


def centred_map(m, X, Y):
    """Retrieval-side centring: subtract the mean of mapped queries / targets
    (estimated on the TRAIN pairs) before cosine -- a classic hub reducer."""
    fx, fy = m
    mq, mt = fx(X).mean(0), fy(Y).mean(0)
    return (lambda x: fx(x) - mq), (lambda y: fy(y) - mt)


def cca_hub_select(X, Y, mode="in", Xu=None, Yu=None, k=10, grid=LAMBDA_GRID):
    """Returns (map, chosen lambda, {lambda: hubness})."""
    scores, maps = {}, {}
    for lam in grid:
        m = C.cca_map(X, Y, lam)
        maps[lam] = m
        if mode == "in":
            scores[lam] = map_hubness(m, X, Y, k)
        else:
            scores[lam] = map_hubness(m, Xu, Yu, k)
    best = min(scores, key=scores.get)
    return maps[best], best, scores


def cca_val_select(X, Y, grid=LAMBDA_GRID):
    return C.tune(lambda a, b, g: C.cca_map(a, b, g), grid, X, Y)
