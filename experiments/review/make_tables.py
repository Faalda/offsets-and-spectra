"""Emit LaTeX rows for the revised paper directly from results/review/*.json (no hand-copied numbers)."""
import json, glob
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results" / "review"
B = {}
for f in glob.glob(str(R / "benchmark_*.json")):
    B.update(json.load(open(f)))
COLS = ["identity", "ortho", "cca", "cca-tuned", "ridge", "rrr", "wridge", "linear-nce", "mlp-nce", "mlp-cos", "lowrank-r8"]
GROUPS = [("e5$\\to$BERT, 17 langs", lambda k: k.startswith("e5/")),
          ("LaBSE$\\to$BERT, 17 langs", lambda k: k.startswith("labse/")),
          ("cross-arch, 7 pairs", lambda k: k.startswith("xa/")),
          ("CLIP (joint), 3 sets", lambda k: k.startswith("clip/")),
          ("Whisper$\\to$XLM-R, 5 splits", lambda k: "libri" in k),
          ("DINOv2$\\to$BERT", lambda k: "dinov2" in k)]
lines = []
for name, sel in GROUPS:
    rs = [v for k, v in B.items() if sel(k)]
    vals = {c: (np.mean([r["p1"][c] for r in rs]) if c in rs[0]["p1"] else None) for c in COLS}
    best = max(v for v in vals.values() if v is not None)
    cells = []
    for c in COLS:
        v = vals[c]
        cells.append("--" if v is None else (f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}"))
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
print("\n".join(lines))
# per-pair supp table
print("\n% ---- per-pair (supp)")
for k, v in sorted(B.items()):
    print(k.replace("_", "\\_").replace("->", "$\\to$") + " & " + " & ".join("--" if c not in v["p1"] else f"{v['p1'][c]:.3f}" for c in COLS)
          + f" & {v['diag']['aniso_unpaired']:.2f} \\\\")
S = json.load(open(R / "selection.json"))["summary"]
print("\n% ---- selection")
for s, m in S["all"].items():
    regs = " & ".join(f"{S[g][s]['mean_regret']:.3f}" for g in ("cross-lingual", "cross-arch", "jointly-trained", "cross-modal"))
    print(f"{s} & {m['acc']*100:.0f} & {m['mean_regret']:.3f} & {m['max_regret']:.3f} & {regs} \\\\")
T = json.load(open(R / "transfer_v2.json"))["_avg"]
print("\n% ---- transfer", {k: round(v, 3) for k, v in T.items()})
