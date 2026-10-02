"""Main mechanism figure: ladder (P@1, hub skew) + trigger scatter (dev filled, held-out open).

Regenerated 2026-10-02 after the Whisper padding-frame correction: the five padded LibriSpeech splits and the two
padding-affected SLURP-Whisper gaps are EXCLUDED from the scatter panels; the padding-corrected SLURP Whisper->XLM-R
gap (reviewer_checks_slurp_corrected.json) and the padding-corrected LibriSpeech gap with dataset-order transcripts
(librispeech_dataset_order_rerun.json) are drawn in panel (a) (no hub statistics exist for them).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results" / "review"
L = json.loads((R / "ladder.json").read_text())["ladder"]
dev = json.loads((R / "decomposition.json").read_text())["rows"]
ho = [r for r in json.loads((R / "heldout_eval.json").read_text())["rows"] if not r["gap"].startswith("slurp/whisper")]
SL = json.loads((R / "reviewer_checks_slurp_corrected.json").read_text())["slurp/whisper-validframes->xlmr"]["p1"]
SLURP_LADDER = [SL["probe"], SL["centred_procrustes"], SL["scaled_procrustes"], SL["cca_selected"]]
LB = json.loads((R / "librispeech_dataset_order_rerun.json").read_text())["mean_p1_over_5_splits"]
LIBRI_LADDER = [LB["probe"], LB["centred_procrustes"], LB["scaled_procrustes"], LB["cca_selected"]]
rungs = ["L0 rotation", "L1 +centring", "L2 +per-dim scaling", "L3 +whitening (CCA)"]
labels = ["rotate", "+centre", "+scale", "+whiten"]
regs = [("text (41)", lambda k: k.startswith(("e5/", "labse/", "xa/")), "#3b6fb6", "o"),
        ("CLIP (3)", lambda k: k.startswith("clip/"), "#7a7a7a", "D"),
        ("image$\\to$text", lambda k: "dinov2" in k, "#1b9e77", "^")]
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5})
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95))
for name, sel, col, mk in regs:
    rs = [v["rungs"] for k, v in L.items() if sel(k)]
    for a, q in ((ax[0], "p1"), (ax[1], "skew")):
        a.plot(range(4), [np.mean([r[x][q] for r in rs]) for x in rungs], marker=mk, color=col, ms=3, lw=1, label=name)
ax[0].plot(range(4), SLURP_LADDER, marker="s", color="#d95f02", ms=3, lw=1, label="speech$\\to$text (SLURP)")
ax[0].plot(range(4), LIBRI_LADDER, marker="v", color="#7b3294", ms=3, lw=1, label="speech$\\to$text (LibriSpeech)")
for a, t, yl in ((ax[0], "(a) P@1 along the ladder", "P@1"), (ax[1], "(b) hubness along the ladder", "$N_1$ skewness")):
    a.set_xticks(range(4)); a.set_xticklabels(labels); a.set_title(t); a.set_ylabel(yl, labelpad=1); a.grid(alpha=.3)
ax[1].set_yscale("log"); pass
devp = [r for r in dev if "libri" not in r["gap"]]
for a, g, x, t, xl in ((ax[2], "g_centre", "mean_dom_x", "(c) centring gain", "source mean dominance"),
                       (ax[3], "g_whiten", "A", "(d) whitening gain", "spectral mismatch $A$")):
    a.scatter([r[x] for r in devp], [r[g] for r in devp], s=7, c="#3b6fb6", label=f"development ({len(devp)})", edgecolors="k", linewidths=.2)
    a.scatter([r[x] for r in ho], [r[g] for r in ho], s=9, facecolors="none", edgecolors="#d95f02", linewidths=.7, label=f"held-out ({len(ho)})")
    a.set_title(t); a.set_xlabel(xl, labelpad=1); a.grid(alpha=.3)
ax[2].legend(fontsize=5, frameon=False, loc="upper left")
fig.tight_layout(pad=0.3, w_pad=0.5, rect=[0, 0.13, 1, 1])
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower left", ncol=5, fontsize=5, frameon=False, bbox_to_anchor=(0.01, 0.0), columnspacing=1.0, handlelength=1.4)
fig.savefig(ROOT / "paper/images/ladder.pdf"); fig.savefig(ROOT / "paper/images/ladder.png", dpi=200); print("ok")
