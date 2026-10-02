"""Score the pre-registered hypotheses (prereg.json) on results/review/heldout.json."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results" / "review"
pre = json.loads((R / "prereg.json").read_text()); H = json.loads((R / "heldout.json").read_text())
rows = []
for k, v in H.items():
    p, d = v["p1"], v["diag"]
    rows.append({"gap": k, "regime": v["regime"], "g_centre": p["rotation+centre"] - p["rotation"],
                 "g_whiten": p["cca-val"] - p["rotation+centre"], "g_csls": p["cca-val+csls"] - p["cca-val"],
                 **{m: d[m] for m in ("mean_dom_x", "mean_dom_y", "A", "ID_train")}, "p1": p, "val": v["val"]})
out = {"n": len(rows), "prereg_sha_of_file": None}
res = {}
for hyp, g, x in (("H1", "g_centre", "mean_dom_x"), ("H2", "g_whiten", "A"), ("H1-exploratory(mean_dom_y)", "g_centre", "mean_dom_y")):
    X = np.array([r[x] for r in rows]); Y = np.array([r[g] for r in rows])
    rho, p = spearmanr(X, Y, alternative="greater")
    res[hyp] = {"rho": float(rho), "p_one_sided": float(p), "n": len(rows), "pass": bool(p < 0.05)}
    # within-family version for the cross-lingual held-out gaps
    for fam in ("mpnet/", "e5/"):
        m = np.array([r["gap"].startswith(fam) for r in rows])
        if m.sum() > 3 and np.std(X[m]) > 0:
            r2, p2 = spearmanr(X[m], Y[m], alternative="greater")
            res[hyp][fam] = {"rho": float(r2), "p_one_sided": float(p2), "n": int(m.sum())}
# H3: selection rule
C = ["identity", "rotation", "rotation+centre", "cca-val", "mlp-nce"]
reg = {"rule": [], "val-select": [], "always-cca-val": [], "always-rotation+centre": []}
for r in rows:
    t = {c: r["p1"][c] for c in C}; best = max(t.values())
    rule = "identity" if r["ID_train"] >= pre["thresholds"]["ID"] else "cca-val"
    vs = max((c for c in C if r["val"].get(c, -1) is not None), key=lambda c: r["val"].get(c, -1))
    for s, c in (("rule", rule), ("val-select", vs), ("always-cca-val", "cca-val"), ("always-rotation+centre", "rotation+centre")):
        reg[s].append(best - t[c])
res["H3"] = {s: {"mean_regret": float(np.mean(v)), "max_regret": float(np.max(v)),
                 "acc": float(np.mean(np.array(v) <= pre["thresholds"]["tie"]))} for s, v in reg.items()}
res["H3"]["pass"] = res["H3"]["rule"]["mean_regret"] <= pre["thresholds"]["H3_max_mean_regret"]
# H4: frozen fits vs frozen dev mean
for g, f in pre["fits"].items():
    X = np.array([r[f["predictor"]] for r in rows]); Y = np.array([r[g] for r in rows])
    mae = float(np.mean(np.abs(f["slope"] * X + f["intercept"] - Y))); mae0 = float(np.mean(np.abs(f["dev_mean"] - Y)))
    res[f"H4|{g}"] = {"mae_fit": mae, "mae_dev_mean": mae0, "pass": mae < mae0}
res["mean_gains"] = {g: float(np.mean([r[g] for r in rows])) for g in ("g_centre", "g_whiten", "g_csls")}
res["mean_p1"] = {c: float(np.mean([r["p1"][c] for r in rows])) for c in list(rows[0]["p1"])}
out["results"] = res; out["rows"] = [{k: v for k, v in r.items() if k not in ("p1", "val")} for r in rows]
(R / "heldout_eval.json").write_text(json.dumps(out, indent=1))
print(json.dumps(res, indent=1))
