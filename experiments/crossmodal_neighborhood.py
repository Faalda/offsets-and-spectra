"""
Neighborhood geometry diagnostics for cross-modal (CLIP) spaces.

For each dataset we compute, on the full test split:
  - P@1 (global Procrustes) -- already known, confirmed here
  - Neighborhood overlap@k  -- what fraction of x_i's k-NN in X are also
                               k-NN of y_i in Y (after cosine normalisation)
  - Trustworthiness          -- do new neighbors in the aligned space
                               belong to the true close neighborhood?
  - Continuity               -- are true neighbors preserved in the aligned space?

If neighborhood overlap is high while Procrustes P@1 is low, the cross-modal
gap is nonlinear but geometrically structured (shared local topology).
If overlap is also low, the spaces share no usable geometry at all.

Uses only cached embeddings -- no downloads needed.
Saves results/crossmodal_neighborhood.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "results" / "emb_cache"
DATASETS  = ["mscoco", "flickr30k", "flickr8k"]
KS        = [5, 10, 20]   # neighbourhood sizes


# ── geometry helpers ──────────────────────────────────────────────────────────

def cosine_norm(X: torch.Tensor) -> torch.Tensor:
    return F.normalize(X.float(), dim=-1)


def knn(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return (n, k) indices of k nearest neighbours (excluding self)."""
    sim = X @ X.T          # (n, n) cosine similarities (X already normalised)
    sim.fill_diagonal_(-2) # exclude self
    return sim.topk(k, dim=1).indices  # (n, k)


def neighborhood_overlap(X: torch.Tensor, Y: torch.Tensor, k: int) -> float:
    """
    Fraction of pairs (x_i, y_i) that share at least one common k-NN:
    |NN_k(x_i) ∩ NN_k(y_i)| / k,  averaged over i.
    """
    nn_x = knn(X, k)  # (n, k)
    nn_y = knn(Y, k)  # (n, k)
    n = X.shape[0]
    overlap = 0.0
    for i in range(n):
        sx = set(nn_x[i].tolist())
        sy = set(nn_y[i].tolist())
        overlap += len(sx & sy) / k
    return overlap / n


def _rank_matrix(X: torch.Tensor) -> torch.Tensor:
    """(n, n) matrix where entry [i,j] = rank of j among i's neighbours (1-indexed, self excluded)."""
    n = X.shape[0]
    sim = X @ X.T
    sim.fill_diagonal_(-2)
    order = sim.argsort(dim=1, descending=True)   # (n, n-1 effectively, but n cols)
    ranks = torch.zeros(n, n, dtype=torch.long)
    row_idx = torch.arange(n).unsqueeze(1).expand_as(order)
    rank_vals = torch.arange(1, n + 1).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_vals)
    return ranks


def trustworthiness(X_orig: torch.Tensor, X_proj: torch.Tensor, k: int) -> float:
    """Score in [0,1]; 1 = no false neighbours introduced in projected space."""
    n = X_orig.shape[0]
    rank_orig = _rank_matrix(X_orig)        # rank of j in i's original neighbourhood
    nn_proj   = knn(X_proj, k)             # (n, k) new neighbours

    # gather original ranks of each projected neighbour
    gathered = rank_orig.gather(1, nn_proj)  # (n, k)
    penalty  = (gathered - k).clamp(min=0).sum().item()
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - norm * penalty)


def continuity(X_orig: torch.Tensor, X_proj: torch.Tensor, k: int) -> float:
    """Score in [0,1]; 1 = no true neighbours lost in projected space."""
    n = X_orig.shape[0]
    rank_proj = _rank_matrix(X_proj)        # rank of j in i's projected neighbourhood
    nn_orig   = knn(X_orig, k)             # (n, k) true neighbours

    gathered = rank_proj.gather(1, nn_orig)  # (n, k)
    penalty  = (gathered - k).clamp(min=0).sum().item()
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - norm * penalty)


def procrustes_p1(Xtr, Ytr, Xte, Yte) -> float:
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    a = cosine_norm(Xte @ R)
    b = cosine_norm(Yte)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print(f"{'dataset':<12} {'P@1':>6}  "
          + "  ".join(f"ovlp@{k}" for k in KS)
          + f"  {'trust@10':>9}  {'cont@10':>8}")

    for name in DATASETS:
        path = CACHE_DIR / f"clip_{name}.pt"
        if not path.exists():
            print(f"{name:<12} cache missing, skipping")
            continue

        cache = torch.load(path, map_location="cpu")
        img = cosine_norm(cache["img"])
        txt = cosine_norm(cache["txt"])
        n   = img.shape[0]

        # use 2/3 train, 1/3 test (same split as crossmodal_clip.py)
        n_tr = min(1000, int(n * 0.67))
        img_tr, txt_tr = img[:n_tr], txt[:n_tr]
        img_te, txt_te = img[n_tr:], txt[n_tr:]

        p1 = procrustes_p1(img_tr, txt_tr, img_te, txt_te)

        # neighborhood metrics computed on test split only
        overlaps = {k: neighborhood_overlap(img_te, txt_te, k) for k in KS}
        trust    = trustworthiness(img_te, txt_te, k=10)
        cont     = continuity(img_te, txt_te, k=10)

        row = dict(dataset=name, p1_procrustes=round(p1, 3),
                   neighborhood_overlap={str(k): round(v, 3) for k, v in overlaps.items()},
                   trustworthiness_k10=round(trust, 3),
                   continuity_k10=round(cont, 3))
        results[name] = row

        ovlp_str = "  ".join(f"{overlaps[k]:.3f}    " for k in KS)
        print(f"{name:<12} {p1:>6.3f}  {ovlp_str}  {trust:>9.3f}  {cont:>8.3f}")

    out = ROOT / "results" / "crossmodal_neighborhood.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")

    # ── interpret ──────────────────────────────────────────────────────────────
    print("\nInterpretation:")
    for name, row in results.items():
        p1   = row["p1_procrustes"]
        ov10 = row["neighborhood_overlap"]["10"]
        t10  = row["trustworthiness_k10"]
        if p1 > 0.5:
            verdict = "globally linearly alignable"
        elif ov10 > 0.3 and t10 > 0.8:
            verdict = "nonlinear but geometrically structured (shared local topology)"
        elif ov10 > 0.15:
            verdict = "partial local structure, no global linear geometry"
        else:
            verdict = "minimal shared geometry"
        print(f"  {name}: P@1={p1:.3f}, overlap@10={ov10:.3f}, trust@10={t10:.3f} => {verdict}")


if __name__ == "__main__":
    main()
