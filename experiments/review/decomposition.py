"""
Mechanistic decomposition of the alignment ladder (response to paper/notes.txt).

Ladder per gap (test P@1, from results/review/hub_dev.json):
  rotation (uncentred x R)  -> + centring  -> + whitening (CCA, val lambda) -> + CSLS
gains:  g_centre = rot+centre - rot ; g_whiten = cca-val - rot+centre ; g_csls = cca-val+csls - cca-val

Candidate label-free predictors (train data only):
  mean_dom   mean dominance of the target space  ||mu_y||^2 / E||y||^2  (and of the source)
  hub_rot    N1 hubness of the uncentred rotation on a validation split
  hub_c      N10 hubness of the CENTRED rotation on the validation split
  A          unpaired spectral mismatch
Question: does each rung have its own measurable trigger?

Usage: python experiments/review/decomposition.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402


def mean_dom(Z):
    return float(Z.mean(0).pow(2).sum() / Z.pow(2).sum(1).mean())


def boot_rho(x, y, B=4000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            v.append(spearmanr(x[i], y[i])[0])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    hd = json.loads((ROOT / "results/review/hub_dev.json").read_text())
    bench = {}
    for f in (ROOT / "results/review").glob("benchmark_*.json"):
        bench.update(json.loads(f.read_text()))
    rows = []
    for group in ("xling_e5", "xling_labse", "xarch", "clip", "xmodal"):
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
            p = hd[name]["p1"]
            a, b, va, vb = C.split_val(Xtr.float(), Ytr.float())
            rc = H.centred_map(C.ortho_map(a, b), a, b)
            rows.append({"gap": name, "regime": regime,
                         "g_centre": p["rotation+centre"] - p["rotation"],
                         "g_whiten": p["cca-val"] - p["rotation+centre"],
                         "g_csls": p["cca-val+csls"] - p["cca-val"],
                         "mean_dom_y": mean_dom(Ytr.float()), "mean_dom_x": mean_dom(Xtr.float()),
                         "hub_rot": hd[name]["diag"]["hub10_rot_val"],
                         "hub_c": H.map_hubness(rc, va, vb, k=10),
                         "A": bench[name]["diag"]["aniso_unpaired"]})
    out = {"rows": rows, "stats": {}}
    # LibriSpeech splits -> one gap for statistics
    lib = [r for r in rows if "libri" in r["gap"]]
    agg = [r for r in rows if "libri" not in r["gap"]] + [{k: (np.mean([r[k] for r in lib]) if not isinstance(lib[0][k], str) else lib[0][k]) for k in lib[0]}]
    sets = {"all": agg, "within": [r for r in agg if r["regime"] in ("cross-lingual", "cross-arch")]}
    for sname, rs in sets.items():
        for g in ("g_centre", "g_whiten", "g_csls"):
            y = np.array([r[g] for r in rs])
            for pr in ("mean_dom_y", "mean_dom_x", "hub_rot", "hub_c", "A"):
                x = np.array([r[pr] for r in rs])
                rho, p = spearmanr(x, y)
                out["stats"][f"{sname}|{g}|{pr}"] = {"n": len(rs), "rho": float(rho), "p": float(p),
                                                    "ci": boot_rho(x, y), "pearson": float(pearsonr(x, y)[0])}
        print(sname, "n", len(rs), "mean gains:", {g: round(float(np.mean([r[g] for r in rs])), 3) for g in ("g_centre", "g_whiten", "g_csls")})
    for k, v in out["stats"].items():
        print(f"{k:<32} rho={v['rho']:+.2f} p={v['p']:.3g} ci=[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] r={v['pearson']:+.2f}")
    (ROOT / "results/review/decomposition.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
