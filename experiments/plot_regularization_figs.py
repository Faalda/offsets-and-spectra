"""
Regenerate the three regularization/geometry figures with the *unified* palette
so that every method keeps the same colour across all paper figures:

    Linear-LS  grey    (#999999)  unregularized baseline
    PLS/Ridge  orange  (#dd8452)  flexible linear (regularized)
    Procrustes blue    (#4c72b0)  rotation  (== ortho in spectrum.py)
    CCA        green   (#55a868)  whitened rotation

Numbers are the confirmed averages from regularized_baselines.py (val-selected
lambda / n_components), reg_baselines2.log:
    ortho 0.848  linls 0.464  ridge 0.705  PLS 0.668  CCA 0.953

Outputs (pdf+png) go straight into paper/images/:
    regularization_geometry_summary, alignment_ladder_hierarchy,
    pairwise_consistency_de_ru_ar
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

IMG = Path(__file__).resolve().parent.parent / "paper" / "images"
IMG.mkdir(parents=True, exist_ok=True)

# ---- unified palette (matches spectrum.py / predictive.py / plot_extra.py) ----
GREY, ORANGE, ORANGE_L, BLUE, GREEN = "#999999", "#dd8452", "#edb48f", "#4c72b0", "#55a868"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(IMG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- (1) regularization_geometry_summary : 5-bar ladder ----------
def summary():
    labels = ["Linear-LS", "PLS", "Ridge", "Procrustes", "CCA"]
    vals   = [0.464, 0.668, 0.705, 0.848, 0.953]
    cols   = [GREY, ORANGE_L, ORANGE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar(range(5), vals, color=cols, edgecolor="k", linewidth=0.5, width=0.72)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
    # delta arrows: regularization (LS->Ridge), geometry (Ridge->Procrustes), whitening (Procrustes->CCA)
    ann = [(2, 0.705, "+0.241"), (3, 0.848, "+0.143"), (4, 0.953, "+0.105")]
    for i, y, txt in ann:
        ax.annotate(txt, xy=(i, y), xytext=(i - 0.72, y + 0.055),
                    fontsize=10, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="k"))
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average P@1"); ax.set_ylim(0, 1.13)
    ax.set_title("Regularization Helps; Geometry Still Wins", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "regularization_geometry_summary")


# ---------- (2) alignment_ladder_hierarchy : boxes + arrows ----------
def ladder():
    boxes = [("Linear-LS\n0.464", GREY),
             ("Ridge / PLS\n0.705 / 0.668", ORANGE),
             ("Procrustes\n0.848", BLUE),
             ("CCA\n0.953", GREEN)]
    steps = ["+0.241\nregularization", "+0.143\ngeometry", "+0.105\nwhitening"]
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4); ax.axis("off")
    bw, bh, y0 = 2.35, 1.15, 0.85
    xs = [0.15, 3.05, 5.95, 8.85]
    for (txt, col), x in zip(boxes, xs):
        box = FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.6, edgecolor="k", facecolor=col)
        ax.add_patch(box)
        ax.text(x + bw / 2, y0 + bh / 2, txt, ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold")
    for i, s in enumerate(steps):
        x = xs[i] + bw
        ax.add_patch(FancyArrowPatch((x + 0.05, y0 + bh / 2), (x + 0.5, y0 + bh / 2),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.6, color="k"))
        ax.text(x + 0.28, y0 + bh + 0.42, s, ha="center", va="bottom", fontsize=10.5)
    ax.set_title("Alignment Ladder: Regularization Helps, Geometry Still Wins",
                 fontsize=14, fontweight="bold", y=1.02)
    save(fig, "alignment_ladder_hierarchy")


# ---------- (3) pairwise_consistency_de_ru_ar : grouped bars ----------
def pairwise():
    pairs = ["de", "ru", "ar"]
    # order: Linear-LS, Ridge, Procrustes, CCA  (from reg_baselines2.log)
    data = {"Linear-LS": [0.667, 0.367, 0.155],
            "Ridge":     [0.842, 0.693, 0.417],
            "Procrustes":[0.933, 0.839, 0.559],
            "CCA":       [0.986, 0.970, 0.756]}
    cols = {"Linear-LS": GREY, "Ridge": ORANGE, "Procrustes": BLUE, "CCA": GREEN}
    x = np.arange(len(pairs)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (name, ys) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, ys, w, label=name, color=cols[name],
               edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=12)
    ax.set_ylabel("P@1"); ax.set_ylim(0, 1.05)
    ax.set_title("Pair-Level Consistency (de, ru, ar)", fontsize=13)
    ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "pairwise_consistency_de_ru_ar")


if __name__ == "__main__":
    summary(); ladder(); pairwise()
