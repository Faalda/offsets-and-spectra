"""
Stage-1 experiment: compare every DSV variant on controllable domain shift.

Protocol
--------
1. Generate source + target domains with a known shift (translation/rotation/both).
2. Train a linear classifier on the SOURCE domain, then freeze it.
3. For each controller: freeze everything except the controller, train it so the
   frozen classifier predicts target labels from transformed target features.
4. Report params and frozen-classifier accuracy on a held-out target split.

Run:
    nadi/.venv/bin/python3 experiments/stage1_synthetic.py --shift both
    nadi/.venv/bin/python3 experiments/stage1_synthetic.py --shift translation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import build_controller, available_controllers           # noqa: E402
from tasks import make_domain_shift                                # noqa: E402


def train_source_classifier(X, y, n_classes, epochs=300, lr=0.05, device="cpu"):
    clf = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(X), y)
        loss.backward()
        opt.step()
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


@torch.no_grad()
def accuracy(clf, ctrl, X, y, device):
    X, y = X.to(device), y.to(device)
    h = ctrl(X) if ctrl is not None else X
    return (clf(h).argmax(-1) == y).float().mean().item()


def train_controller(name, clf, data, epochs, lr, device, **ctrl_kwargs):
    ctrl = build_controller(name, dim=data.dim, n_layers=1, **ctrl_kwargs).to(device)
    Xt, yt = data.Xt_train.to(device), data.yt_train.to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(ctrl(Xt)), yt)
        loss.backward()
        opt.step()
    ctrl.eval()
    return ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shift", choices=["translation", "rotation", "both", "none"], default="both")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--controllers", nargs="*", default=available_controllers())
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data = make_domain_shift(n_classes=args.classes, dim=args.dim,
                             shift=args.shift, seed=args.seed)
    clf = train_source_classifier(data.Xs_train, data.ys_train, data.n_classes, device=device)

    src_acc = accuracy(clf, None, data.Xs_train, data.ys_train, device)
    none_acc = accuracy(clf, None, data.Xt_test, data.yt_test, device)

    print(f"\nShift = {args.shift} | dim = {args.dim} | classes = {args.classes}")
    print(f"Source-domain accuracy (sanity)      : {src_acc:6.3f}")
    print(f"Target accuracy, no controller       : {none_acc:6.3f}")
    print("-" * 58)
    print(f"{'controller':<12}{'params':>10}{'target_acc':>14}")
    print("-" * 58)

    rows = []
    for name in args.controllers:
        ctrl = train_controller(name, clf, data, args.epochs, args.lr, device)
        acc = accuracy(clf, ctrl, data.Xt_test, data.yt_test, device)
        rows.append((name, ctrl.num_params(), acc))

    for name, params, acc in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<12}{params:>10,}{acc:>14.3f}")
    print("-" * 58)


if __name__ == "__main__":
    main()
