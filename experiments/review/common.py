"""
Shared alignment maps + protocol for the AAAI-27 review-response experiments.

Every map is fitted on TRAIN pairs only; any hyperparameter is selected on a
VALIDATION split carved from TRAIN (never on test); the selected configuration is
then refitted on the full TRAIN set and evaluated once on TEST.

A "map" returns a pair of closures (fx, fy) that send source / target embeddings
into a common comparison space; retrieval is cosine P@1 of fx(x_i) against
{fy(y_j)}.  For maps into the target space fy is the identity.

    identity        fx=x, fy=y                        (needs equal dims)
    ortho           fx=xR, R=UV^T from SVD(Xc^T Yc)   (semi-orthogonal if d_x != d_y)
    cca(lam)        fx=(x-mx)Cxx^-1/2 U, fy=(y-my)Cyy^-1/2 V,
                    U,V from SVD(Cxx^-1/2 Cxy Cyy^-1/2), C.. = cov + lam*I
                    == whiten both sides, then orthogonal Procrustes between them
    ridge(lam)      fx=(x-mx)W+my, W=(Xc^T Xc + lam n I)^-1 Xc^T Yc
    rrr(k,lam)      reduced-rank ridge: W_k = W V_k V_k^T (V: right sing. vecs of Xc W)
    wridge(lam)     whiten both sides (same whitening as cca), ridge between the
                    whitened spaces, compare in whitened target space
                    -> isolates the orthogonality constraint from whitening
    ortho_wmetric   ortho map, then compare after target whitening
                    -> isolates the retrieval-metric effect of whitening
    mlp             2-layer residual MLP, AdamW weight decay + dropout, early
                    stopping; loss in {cosine, infonce}; grid chosen on validation
    linear_nce      linear map trained with InfoNCE (objective control for mlp)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

CCA_LAM = 0.1
LAMBDAS_RIDGE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LAMBDAS_CCA = [1e-3, 1e-2, 1e-1, 1.0]
RRR_RANKS = [16, 32, 64, 128, 256, 512]


# ------------------------------------------------------------------ metrics
def p1_vec(A, B):
    a, b = F.normalize(A.float(), dim=-1), F.normalize(B.float(), dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0], device=a.device)).float()


def p1(A, B):
    return p1_vec(A, B).mean().item()


def evalmap(m, X, Y):
    fx, fy = m
    return p1(fx(X), fy(Y))


def split_val(X, Y, frac=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    nv = max(64, int(round(X.shape[0] * frac)))
    v, t = perm[:nv], perm[nv:]
    return X[t], Y[t], X[v], Y[v]


# ------------------------------------------------------------------ closed forms
def _isqrt(C):
    w, V = torch.linalg.eigh(C)
    return (V * w.clamp(min=1e-8).rsqrt()) @ V.T


def identity_map(X, Y):
    return (lambda x: x), (lambda y: y)


def ortho_map(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return (lambda x: x @ R), (lambda y: y)


def whiteners(X, Y, lam):
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    Cxx = Xc.T @ Xc / n + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    Cyy = Yc.T @ Yc / n + lam * torch.eye(Y.shape[1], dtype=Y.dtype, device=Y.device)
    return mx, my, _isqrt(Cxx), _isqrt(Cyy), Xc, Yc


def cca_map(X, Y, lam=CCA_LAM):
    mx, my, Mx, My, Xc, Yc = whiteners(X, Y, lam)
    Cxy = Xc.T @ Yc / X.shape[0]
    U, _, Vt = torch.linalg.svd(Mx @ Cxy @ My, full_matrices=False)
    Wx, Wy = Mx @ U, My @ Vt.T
    return (lambda x: (x - mx) @ Wx), (lambda y: (y - my) @ Wy)


def ridge_W(Xc, Yc, lam):
    d = Xc.shape[1]
    A = Xc.T @ Xc + lam * Xc.shape[0] * torch.eye(d, dtype=Xc.dtype, device=Xc.device)
    return torch.linalg.solve(A, Xc.T @ Yc)


def ridge_map(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    W = ridge_W(X - mx, Y - my, lam)
    return (lambda x: (x - mx) @ W + my), (lambda y: y)


def linls_map(X, Y):
    return ridge_map(X, Y, 1e-10)


def rrr_map(X, Y, k, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    W = ridge_W(Xc, Y - my, lam)
    _, _, Vt = torch.linalg.svd(Xc @ W, full_matrices=False)
    Vk = Vt[:k].T
    Wk = W @ Vk @ Vk.T
    return (lambda x: (x - mx) @ Wk + my), (lambda y: y)


def wridge_map(X, Y, lam, cca_lam=CCA_LAM):
    mx, my, Mx, My, Xc, Yc = whiteners(X, Y, cca_lam)
    Xw, Yw = Xc @ Mx, Yc @ My
    W = ridge_W(Xw, Yw, lam)
    return (lambda x: ((x - mx) @ Mx) @ W), (lambda y: (y - my) @ My)


def ortho_wmetric_map(X, Y, cca_lam=CCA_LAM):
    fx, _ = ortho_map(X, Y)
    mx, my, Mx, My, _, _ = whiteners(X, Y, cca_lam)
    return (lambda x: (fx(x) - my) @ My), (lambda y: (y - my) @ My)


def lowrank_closed(X, Y, r):
    """Closed-form rank-r residual x -> x + x W_r (LoRA hypothesis class, MSE fit)."""
    W = torch.linalg.lstsq(X, Y - X).solution
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    return (lambda x: x + x @ Wr), (lambda y: y)


# ------------------------------------------------------------------ tuned wrappers
def tune(fit_fn, grid, Xtr, Ytr, seed=0):
    """Select grid point on a validation split of TRAIN; refit on full TRAIN."""
    a, b, va, vb = split_val(Xtr, Ytr, seed=seed)
    scores = {g: evalmap(fit_fn(a, b, g), va, vb) for g in grid}
    best = max(scores, key=scores.get)
    return fit_fn(Xtr, Ytr, best), best, scores


def ridge_tuned(X, Y):
    return tune(lambda a, b, g: ridge_map(a, b, g), LAMBDAS_RIDGE, X, Y)


def cca_tuned(X, Y):
    return tune(lambda a, b, g: cca_map(a, b, g), LAMBDAS_CCA, X, Y)


def rrr_tuned(X, Y):
    kmax = min(X.shape[1], Y.shape[1])
    grid = [(k, l) for k in RRR_RANKS if k <= kmax for l in (1e-4, 1e-2, 1e-1, 1.0)]
    return tune(lambda a, b, g: rrr_map(a, b, *g), grid, X, Y)


def wridge_tuned(X, Y):
    return tune(lambda a, b, g: wridge_map(a, b, g), LAMBDAS_RIDGE, X, Y)


# ------------------------------------------------------------------ trained maps
class ResMLP(torch.nn.Module):
    def __init__(self, dx, dy, h, p):
        super().__init__()
        self.lin = torch.nn.Linear(dx, dy)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dx, h), torch.nn.GELU(),
                                       torch.nn.Dropout(p), torch.nn.Linear(h, dy))
        torch.nn.init.zeros_(self.mlp[-1].weight); torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


def _loss(pred, y, kind, tau=0.05):
    if kind == "cosine":
        return (1 - F.cosine_similarity(pred, y, dim=-1)).mean()
    a, b = F.normalize(pred, dim=-1), F.normalize(y, dim=-1)
    logits = a @ b.T / tau
    t = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))


def train_net(Xtr, Ytr, Xva, Yva, arch, h, p, wd, loss, lr=1e-3, epochs=300,
              bs=256, seed=0, patience=30, fixed_epochs=None, device="cuda:0"):
    torch.manual_seed(seed)
    dx, dy = Xtr.shape[1], Ytr.shape[1]
    net = (ResMLP(dx, dy, h, p) if arch == "mlp" else torch.nn.Linear(dx, dy)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    mx = Xtr.mean(0)
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    n_ep = fixed_epochs or epochs
    for ep in range(1, n_ep + 1):
        net.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            _loss(net(Xtr[idx] - mx), Ytr[idx], loss).backward()
            opt.step()
        if fixed_epochs is None and (ep % 5 == 0):
            net.eval()
            with torch.no_grad():
                s = p1(net(Xva.to(device) - mx), Yva.to(device))
            if s > best:
                best, best_ep, bad = s, ep, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def fx(x):
        with torch.no_grad():
            return net(x.to(device) - mx).cpu()
    return (fx, lambda y: y), best, best_ep


MLP_GRID = [(h, p, wd) for h in (512, 2048) for p in (0.0, 0.3) for wd in (1e-4, 1e-2, 1e-1)]


def mlp_tuned(X, Y, loss="infonce", arch="mlp", seed=0, device="cuda:0"):
    """Grid + early stopping on validation; refit on full TRAIN for the chosen #epochs."""
    a, b, va, vb = split_val(X, Y, seed=seed)
    grid = MLP_GRID if arch == "mlp" else [(0, 0.0, wd) for wd in (1e-4, 1e-2, 1e-1)]
    res = {}
    for g in grid:
        _, s, ep = train_net(a, b, va, vb, arch, *g, loss=loss, seed=seed, device=device)
        res[g] = (s, ep)
    best = max(res, key=lambda g: res[g][0])
    m, _, _ = train_net(X, Y, None, None, arch, *best, loss=loss, seed=seed,
                        fixed_epochs=max(5, res[best][1]), device=device)
    return m, {"h": best[0], "dropout": best[1], "wd": best[2], "epochs": res[best][1],
               "val_p1": res[best][0]}, {str(k): v for k, v in res.items()}


# ------------------------------------------------------------------ diagnostics
def norm_spectrum(Z, k=None):
    Zc = Z - Z.mean(0)
    ev = torch.linalg.eigvalsh(Zc.T @ Zc / Z.shape[0]).clamp(min=0)
    ev = ev.sort(descending=True).values
    if k is not None:
        ev = ev[:k]
    return ev / ev.sum()


def anisotropy(X, Y):
    """L1 distance between normalised covariance eigen-spectra. Needs NO pairing:
    each spectrum is computed from its own space alone.  Unequal dims: truncate
    to the smaller dimension (spectral mass beyond it is ~0 for d<=n)."""
    k = min(X.shape[1], Y.shape[1])
    return (norm_spectrum(X, k) - norm_spectrum(Y, k)).abs().sum().item()


def self_anisotropy(Z):
    """Deviation of a single space from isotropy: L1 distance of its normalised
    spectrum to the uniform spectrum (0 = isotropic)."""
    s = norm_spectrum(Z)
    return (s - 1.0 / s.numel()).abs().sum().item()


@dataclass
class Timer:
    t: dict = field(default_factory=dict)

    def __call__(self, name, fn, *a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter(); out = fn(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t[name] = time.perf_counter() - t0
        return out
