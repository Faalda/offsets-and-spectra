"""Summarise results/review/benchmark_*.json: per-group means, win counts, CCA-vs-best."""
import json, glob, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
B = {}
for f in sorted(glob.glob(str(ROOT / "results/review/benchmark_*.json"))):
    B[Path(f).stem.replace("benchmark_", "")] = json.load(open(f))
M = ["identity", "ortho", "cca", "cca-tuned", "linear-ls", "ridge", "rrr", "wridge", "mlp-cos", "mlp-nce", "linear-nce", "lowrank-r8"]
for g, d in B.items():
    rows = list(d.values())
    if g == "xmodal":  # average the 5 LibriSpeech seeds into one gap
        lib = [r for k, r in d.items() if "libri" in k]
        print(f"[{g}] libri 5 seeds mean/std:", {m: (round(np.mean([r['p1'][m] for r in lib]), 3), round(np.std([r['p1'][m] for r in lib]), 3)) for m in M if m in lib[0]['p1']})
    print(f"[{g}] n={len(rows)}  " + "  ".join(f"{m}={np.mean([r['p1'][m] for r in rows]):.3f}" for m in M if m in rows[0]["p1"]))
    best_other = [max(v for k, v in r["p1"].items() if k not in ("cca", "cca-tuned")) for r in rows]
    cca = [r["p1"]["cca"] for r in rows]
    wins = sum(c >= b - 1e-9 for c, b in zip(cca, best_other))
    print(f"     cca(lam=.1) >= every non-CCA map in {wins}/{len(rows)}; mean(cca - best non-CCA) = {np.mean(np.array(cca)-np.array(best_other)):+.3f}")
    who = [max(((k, v) for k, v in r["p1"].items() if k not in ("cca", "cca-tuned")), key=lambda t: t[1])[0] for r in rows]
    print("     best non-CCA map:", {w: who.count(w) for w in set(who)})
    au, ap = [r["diag"]["aniso_unpaired"] for r in rows], [r["diag"]["aniso_paired_half"] for r in rows]
    print(f"     |A_unpaired - A_paired_half| mean={np.mean(np.abs(np.array(au)-np.array(ap))):.4f} max={np.max(np.abs(np.array(au)-np.array(ap))):.4f}; A mean={np.mean(ap):.3f}")
    print("     mean fit time (s):", {k: round(np.mean([r['time_s'][k] for r in rows]), 3) for k in rows[0]["time_s"]})
