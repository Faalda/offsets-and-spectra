"""
Dose-response ladder (notes.txt #3) + small-probe prognosis (notes.txt #5).

LADDER, same fitted correspondence wherever possible. For each gap:
  L0 rotation            x R                       (R from centred Procrustes, uncentred retrieval)
  L1 + centring          (x R - m_q), (y - m_t)    SAME R -> an intervention on the offset only
  L2 + per-dim scaling   standardise each coordinate of both sides, then Procrustes
  L3 + whitening         CCA (val lambda)          full covariance correction
  L4 + CSLS              retrieval-side hub correction on top of L3
Per rung: P@1, N1 hub skew, top-1% target share, % targets never retrieved,
and KL(retrieved-target distribution || uniform)  ("concentration excess", notes #4).

PROBE: compute the diagnostics (ID, mean dominance, A) and the selector's decision from only
m = 50/100/250 training pairs and ask whether the decision matches the one made with all pairs,
and the regret of the probe's decision when the map is then fitted on all pairs.
Usage: python experiments/review/ladder.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402
from decomposition import mean_dom  # noqa: E402


def hubstats(m, X, Y):
    fx, fy = m
    S = F.normalize(fx(X).float(), dim=-1) @ F.normalize(fy(Y).float(), dim=-1).T
    nn = S.argmax(1); n = S.shape[1]
    k = torch.bincount(nn, minlength=n).double().numpy(); hs = np.sort(k)[::-1]
    p = k / k.sum(); nz = p > 0
    return {"p1": float((nn == torch.arange(len(nn))).float().mean()),
            "skew": float(skew(k)) if k.std() > 0 else 0.0,
            "top1pct": float(hs[:max(1, n // 100)].sum() / len(nn)),
            "never": float((k == 0).mean()),
            "kl_uniform": float((p[nz] * np.log(p[nz] * n)).sum())}


def std_map(X, Y):
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-8, Y.mean(0), Y.std(0) + 1e-8
    Xs, Ys = (X - mx) / sx, (Y - my) / sy
    U, _, Vt = torch.linalg.svd(Xs.T @ Ys, full_matrices=False); R = U @ Vt
    return (lambda x: ((x - mx) / sx) @ R), (lambda y: (y - my) / sy)


def identity_subset_full_gallery(Xq, Yfull, query_indices):
    """Identity P@1 for sampled queries against the selector's full target gallery."""
    S = F.normalize(Xq.float(), dim=-1) @ F.normalize(Yfull.float(), dim=-1).T
    return (S.argmax(1) == query_indices).float().mean().item()


def main():
    out = {"ladder": {}, "probe": {}}
    g = torch.Generator().manual_seed(0)
    for group in ("xling_e5", "xling_labse", "xarch", "clip", "xmodal"):
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
            Xtr, Ytr, Xte, Yte = (t.float() for t in (Xtr, Ytr, Xte, Yte))
            rot = C.ortho_map(Xtr, Ytr); ccav, lv, _ = H.cca_val_select(Xtr, Ytr)
            L = {"L0 rotation": hubstats(rot, Xte, Yte),
                 "L1 +centring": hubstats(H.centred_map(rot, Xtr, Ytr), Xte, Yte),
                 "L2 +per-dim scaling": hubstats(std_map(Xtr, Ytr), Xte, Yte),
                 "L3 +whitening (CCA)": hubstats(ccav, Xte, Yte)}
            L["L4 +CSLS"] = {"p1": H.csls_p1(ccav, Xte, Yte)}
            out["ladder"][name] = {"regime": regime, "rungs": L}
            # probe
            same = Xtr.shape[1] == Ytr.shape[1]
            full_dec = "identity" if same and C.p1(Xtr, Ytr) >= 0.5 else "cca"
            pr = {}
            for m in (50, 100, 250):
                idx = torch.randperm(Xtr.shape[0], generator=g)[:m]
                a, b = Xtr[idx], Ytr[idx]
                # Keep the full target gallery. Retrieval within an m-item gallery
                # changes chance performance and is not comparable to ID_train.
                id_small = identity_subset_full_gallery(a, Ytr, idx) if same else 0.0
                dec = "identity" if same and id_small >= 0.5 else "cca"
                h = m // 2
                pr[m] = {"decision": dec, "matches_full": dec == full_dec,
                         "ID_full_gallery": id_small,
                          "mean_dom_x": mean_dom(a), "A": C.anisotropy(a[:h], b[h:2 * h])}
            test = {"identity": C.evalmap(C.identity_map(Xtr, Ytr), Xte, Yte) if same else -1.0,
                    "cca": C.evalmap(ccav, Xte, Yte)}
            for m in pr:
                pr[m]["regret"] = max(test.values()) - test[pr[m]["decision"]]
            pr["full"] = {"mean_dom_x": mean_dom(Xtr), "A": C.anisotropy(Xtr[:Xtr.shape[0] // 2], Ytr[Xtr.shape[0] // 2:])}
            out["probe"][name] = pr
            print(name, {k: round(v["p1"], 3) for k, v in L.items()}, {m: pr[m]["decision"] for m in (50, 100, 250)}, flush=True)
    (ROOT / "results/review/ladder.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
