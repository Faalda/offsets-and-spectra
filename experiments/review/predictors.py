"""
R2-W1 / AI-W3: which label-free statistic predicts the whitening benefit
Delta = P@1(cca) - P@1(ortho) on the within-modality gaps (and where does it fail)?

Candidates (all computed on TRAIN data only):
  A            spectral mismatch between the two spaces (unpaired halves)
  self_x/y     each space's own distance from isotropy
  hub_rot      N1 k-occurrence skewness of the rotation's retrieval on a validation
               split (rotation fitted on the other 80%): no test labels used
Reports Pearson/Spearman with bootstrap CIs, pooled and per encoder family,
and leave-one-family-out prediction error of a 1-D linear fit.
Also regenerates the Delta-vs-A figure with correct n (fixes AI-W3 annotation bug).

Usage: python experiments/review/predictors.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, skew

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
from benchmark import gaps  # noqa: E402


def hub_skew(fx, fy, X, Y):
    a, b = F.normalize(fx(X), dim=-1), F.normalize(fy(Y), dim=-1)
    nn = (a @ b.T).argmax(1)
    k = torch.bincount(nn, minlength=b.shape[0]).float().numpy()
    return float(skew(k)) if k.std() > 0 else 0.0   # perfect 1-to-1 retrieval = no hubs


def boot(x, y, fn, B=5000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            v.append(fn(x[i], y[i])[0])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None


def main():
    bench = {}
    for f in (ROOT / "results" / "review").glob("benchmark_*.json"):
        bench.update(json.loads(f.read_text()))
    rows = []
    for group in ("xling_e5", "xling_labse", "xarch"):
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
            b = bench[name]
            a, bb, va, vb = C.split_val(Xtr, Ytr)
            rows.append({"gap": name, "family": name.split("/")[0], "delta": b["p1"]["cca"] - b["p1"]["ortho"],
                         "A": b["diag"]["aniso_unpaired"], "self_x": b["diag"]["self_aniso_x"],
                         "self_y": b["diag"]["self_aniso_y"],
                         "hub_rot": hub_skew(*C.ortho_map(a, bb), va, vb),
                         "hub_cca": hub_skew(*C.cca_map(a, bb), va, vb)})
            print(rows[-1], flush=True)
    D = np.array([r["delta"] for r in rows])
    out = {"rows": rows, "stats": {}}
    for k in ("A", "self_x", "self_y", "hub_rot"):
        x = np.array([r[k] for r in rows]); st = {}
        for fam in ("all", "e5", "labse", "xa"):
            m = np.ones(len(rows), bool) if fam == "all" else np.array([r["family"] == fam for r in rows])
            if np.std(x[m]) == 0:
                st[fam] = None; continue
            pr, pp = pearsonr(x[m], D[m]); sr, sp = spearmanr(x[m], D[m])
            st[fam] = {"n": int(m.sum()), "pearson": float(pr), "p": float(pp), "pearson_ci": boot(x[m], D[m], pearsonr),
                       "spearman": float(sr), "spearman_p": float(sp), "spearman_ci": boot(x[m], D[m], spearmanr)}
        # leave-one-family-out MAE of a 1-D linear fit vs predicting the training mean
        err, err0 = [], []
        for fam in ("e5", "labse", "xa"):
            te = np.array([r["family"] == fam for r in rows]); tr = ~te
            c = np.polyfit(x[tr], D[tr], 1) if np.std(x[tr]) > 0 else np.array([0.0, D[tr].mean()])
            err += list(np.abs(np.polyval(c, x[te]) - D[te])); err0 += list(np.abs(D[tr].mean() - D[te]))
        st["lofo_mae"], st["lofo_mae_mean_baseline"] = float(np.mean(err)), float(np.mean(err0))
        out["stats"][k] = st
        print(k, json.dumps(st))
    (ROOT / "results" / "review" / "predictors.json").write_text(json.dumps(out, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9), sharey=True)
    sty = {"e5": ("o", "#3b6fb6", "e5$\\to$BERT"), "labse": ("s", "#d95f02", "LaBSE$\\to$BERT"), "xa": ("^", "#1b9e77", "cross-arch")}
    for ax, k, lab in ((axs[0], "A", "spectral mismatch $A$"), (axs[1], "hub_rot", "rotation hubness (skew)")):
        for fam, (mk, col, nm) in sty.items():
            rs = [r for r in rows if r["family"] == fam]
            ax.scatter([r[k] for r in rs], [r["delta"] for r in rs], marker=mk, c=col, s=9, label=f"{nm} ({len(rs)})", edgecolors="k", linewidths=.25)
        s = out["stats"][k]["all"]
        ax.set_title(f"$\\rho{{=}}{s['spearman']:.2f}$ [{s['spearman_ci'][0]:.2f},{s['spearman_ci'][1]:.2f}]", fontsize=6.5, pad=2)
        ax.set_xlabel(lab, fontsize=6.5, labelpad=1); ax.tick_params(labelsize=5.5, pad=1); ax.grid(alpha=.3)
    axs[0].set_ylabel("$\\Delta$ = CCA $-$ rotation", fontsize=6.5, labelpad=1)
    axs[0].legend(fontsize=5, loc="upper right", frameon=False, handletextpad=.2, borderaxespad=.2)
    fig.tight_layout(pad=0.3, w_pad=0.4); fig.savefig(ROOT / "paper" / "images" / "delta_predictors.pdf"); fig.savefig(ROOT / "paper" / "images" / "delta_predictors.png", dpi=200); print("fig saved")


if __name__ == "__main__":
    main()
