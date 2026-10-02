"""Score prereg2.json (COCO + sentiment) and prereg3.json (Laya / Qwen3) exactly as frozen."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results/review"
coco = json.loads((R / "coco_gaps.json").read_text()); laya = json.loads((R / "laya_gaps.json").read_text())
sent = json.loads((R / "sentiment.json").read_text())
CANDS = ["identity", "rotation", "rotation+centre", "cca-val", "mlp-nce"]


def regret(gs):
    out = []
    for v in gs.values():
        t = {c: v["p1"][c] for c in CANDS}; best = max(t.values())
        pick = "identity" if v["diag"]["ID_train"] >= 0.5 else "cca-val"
        out.append(best - t[pick])
    return float(np.mean(out)), float(np.max(out))


res = {}
it = {k: v for k, v in coco.items() if v["regime"] == "image-text"}
ii = {k: v for k, v in coco.items() if v["regime"] == "image-image"}
res["P1 image->text CCA<0.5 on every gap"] = {"values": {k: round(v["p1"]["cca-val"], 3) for k, v in it.items()},
                                             "pass": all(v["p1"]["cca-val"] < 0.5 for v in it.values())}
res["P2 hub skew after CCA >2 on majority (image->text)"] = {"values": {k: round(v["hub"]["cca"], 2) for k, v in it.items()},
                                                            "pass": sum(v["hub"]["cca"] > 2 for v in it.values()) > len(it) / 2}
res["P3 image->image CCA>0.8 and centred skew<1.5 on majority"] = {
    "values": {k: (round(v["p1"]["cca-val"], 3), round(v["hub"]["centred"], 2)) for k, v in ii.items()},
    "pass": sum(v["p1"]["cca-val"] > 0.8 and v["hub"]["centred"] < 1.5 for v in ii.values()) > len(ii) / 2}
m, mx = regret(coco); res["P4 rule regret<=0.01 (COCO batch)"] = {"mean": m, "max": mx, "pass": m <= 0.01}
res["P5 sentiment: rho(P@1,acc)<0.5 AND best map in {centred rotation, CCA-k}"] = {
    "rho": sent["_P5_spearman_p1_vs_acc"], "best": sent["_best_map"],
    "pass": sent["_P5_spearman_p1_vs_acc"] < 0.5 and sent["_best_map"] in ("ortho-centred", "cca-k")}
ind = {"bert": ["dinov2", "vit", "convnext"], "e5": ["dinov2", "vit", "convnext"], "mpnet": ["dinov2", "vit", "convnext"]}
p6 = {}
for t, srcs in ind.items():
    best_ind = max(coco[f"coco/{s}->{t}"]["p1"]["cca-val"] for s in srcs)
    for l in ("laya-tower", "laya-connector"):
        p6[f"{l}->{t}"] = (round(laya[f"laya/{l}->{t}"]["p1"]["cca-val"], 3), round(best_ind, 3))
res["P6 language-trained image encoders beat every independent one (per text target)"] = {
    "values(laya, best independent)": p6, "pass": all(a > b for a, b in p6.values())}
ft = laya["laya/modernbert-base->laya-modernbert"]
res["P7 fine-tune gap ID>=0.5"] = {"ID": ft["diag"]["ID_train"], "pass": ft["diag"]["ID_train"] >= 0.5}
p8 = {}
for k in ("laya/laya-modernbert->bert", "laya/laya-modernbert->e5", "laya/qwen3->bert", "laya/qwen3->e5"):
    p = laya[k]["p1"]; frac = (p["rotation+centre"] - p["rotation"]) / max(1e-9, p["cca-val"] - p["rotation"])
    p8[k] = {"cca": round(p["cca-val"], 3), "centring_share": round(frac, 3), "ok": p["cca-val"] >= 0.9 and frac >= 0.8}
res["P8 text pattern (CCA>=0.9, centring>=80% of gain) on majority"] = {"values": p8, "pass": sum(v["ok"] for v in p8.values()) > len(p8) / 2}
m, mx = regret(laya); res["P9 rule regret<=0.01 (Laya batch)"] = {"mean": m, "max": mx, "pass": m <= 0.01}
(R / "prereg23_eval.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(("PASS " if v["pass"] else "FAIL ") + k); print("     ", {a: b for a, b in v.items() if a != "pass"})
