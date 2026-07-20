"""
Plot the Stage-1 params-vs-accuracy Pareto curve from harden_stage1.py output.

    nadi/.venv/bin/python3 experiments/plot_pareto.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(path):
    fams = defaultdict(lambda: ([], [], []))
    with open(path) as f:
        for r in csv.DictReader(f):
            p, m, s = fams[r["family"]]
            p.append(int(r["params"])); m.append(float(r["mean"])); s.append(float(r["std"]))
    return fams


def main():
    csv_path = RESULTS / "stage1_pareto.csv"
    if not csv_path.exists():
        sys.exit("Run harden_stage1.py first to produce stage1_pareto.csv")
    fams = load(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    styles = {"ortho": ("o-", "#1f77b4", "OrthoDSV (Householder)"),
              "lowrank": ("s--", "#d62728", "LoRA (low-rank)")}
    for fam, (params, mean, std) in fams.items():
        order = sorted(range(len(params)), key=lambda i: params[i])
        params = [params[i] for i in order]; mean = [mean[i] for i in order]; std = [std[i] for i in order]
        fmt, color, label = styles.get(fam, ("o-", "gray", fam))
        ax.errorbar(params, mean, yerr=std, fmt=fmt, color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.8)

    ax.set_xscale("log")
    ax.set_xlabel("adapter parameters (log scale)")
    ax.set_ylabel("target accuracy (5 seeds)")
    ax.set_title("Stage-1 Pareto: orthogonal alignment vs low-rank\n(rotation + translation shift)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(RESULTS / f"stage1_pareto.{ext}", dpi=150)
    # also drop a vector PDF next to the paper source
    paper_dir = Path(__file__).resolve().parent.parent / "paper"
    if paper_dir.exists():
        fig.savefig(paper_dir / "stage1_pareto.pdf")
    print(f"Wrote stage1_pareto.{{png,svg,pdf}} to {RESULTS}/ and paper/")


if __name__ == "__main__":
    main()
