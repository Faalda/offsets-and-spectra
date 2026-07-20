"""
Hardened Stage-1: multi-seed error bars + a params-vs-accuracy Pareto sweep.

Two reports:
  1. ROBUSTNESS — every controller × every shift type, mean ± std over N seeds.
  2. PARETO — ortho (Householder, varying K) vs lowrank (varying rank): accuracy
     as a function of parameter budget, on the rotation+translation shift.

Writes CSV to results/ for plotting. Run:
    nadi/.venv/bin/python3 experiments/harden_stage1.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import build_controller                                   # noqa: E402
from tasks import make_domain_shift                                # noqa: E402
from experiments.stage1_synthetic import (                        # noqa: E402
    train_source_classifier, train_controller, accuracy,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return m, v ** 0.5


def run_one(name, shift, seed, dim, classes, epochs, lr, device, **kw):
    torch.manual_seed(seed)
    data = make_domain_shift(n_classes=classes, dim=dim, shift=shift, seed=seed)
    clf = train_source_classifier(data.Xs_train, data.ys_train, data.n_classes, device=device)
    ctrl = train_controller(name, clf, data, epochs, lr, device, **kw)
    return accuracy(clf, ctrl, data.Xt_test, data.yt_test, device), ctrl.num_params()


def robustness(args, device):
    controllers = ["dsv", "matrix", "gated", "mixture", "lowrank", "ortho", "affine"]
    shifts = ["translation", "rotation", "both"]
    print(f"\n=== ROBUSTNESS ({args.seeds} seeds, dim={args.dim}, classes={args.classes}) ===")
    print(f"{'controller':<12}" + "".join(f"{s:>20}" for s in shifts))
    rows = []
    for name in controllers:
        cells = []
        for shift in shifts:
            accs = [run_one(name, shift, s, args.dim, args.classes, args.epochs, args.lr, device)[0]
                    for s in range(args.seeds)]
            m, sd = mean_std(accs)
            cells.append(f"{m:.3f}±{sd:.3f}")
            rows.append({"controller": name, "shift": shift, "mean": f"{m:.4f}", "std": f"{sd:.4f}"})
        print(f"{name:<12}" + "".join(f"{c:>20}" for c in cells))
    with open(RESULTS / "stage1_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["controller", "shift", "mean", "std"])
        w.writeheader(); w.writerows(rows)


def pareto(args, device):
    print(f"\n=== PARETO: params vs accuracy on 'both' ({args.seeds} seeds, dim={args.dim}) ===")
    print(f"{'method':<22}{'params':>10}{'accuracy':>16}")
    rows = []
    configs = ([("ortho", {"parametrization": "householder", "n_reflections": k}, f"ortho-hh K={k}")
                for k in (1, 2, 4, 8, 16)]
               + [("lowrank", {"rank": r}, f"lowrank r={r}") for r in (1, 2, 4, 8, 16)])
    for name, kw, label in configs:
        out = [run_one(name, "both", s, args.dim, args.classes, args.epochs, args.lr, device, **kw)
               for s in range(args.seeds)]
        accs = [a for a, _ in out]
        params = out[0][1]
        m, sd = mean_std(accs)
        print(f"{label:<22}{params:>10,}{f'{m:.3f}±{sd:.3f}':>16}")
        rows.append({"method": label, "family": name, "params": params,
                     "mean": f"{m:.4f}", "std": f"{sd:.4f}"})
    with open(RESULTS / "stage1_pareto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "family", "params", "mean", "std"])
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(exist_ok=True)
    robustness(args, device)
    pareto(args, device)
    print(f"\nCSVs written to {RESULTS}/")


if __name__ == "__main__":
    main()
