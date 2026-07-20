"""
Stage-4 experiment: cross-modal speech↔text alignment (SLURP-TN).

The method's predicted sweet spot. Align frozen wav2vec2 speech embeddings to
frozen XLM-R text embeddings; evaluate cross-modal retrieval P@1 against the
analytic Procrustes (orthogonal) and least-squares (linear) optima. `lowrank`
here is LoRA-used-as-an-alignment-map — the direct LoRA comparison.

Run:
    nadi/.venv/bin/python3 experiments/stage4_crossmodal.py --warmstart-ortho
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import available_controllers                              # noqa: E402
from dsv.baselines import procrustes, linear_ls                   # noqa: E402
from tasks.crossmodal import load_crossmodal                      # noqa: E402
from experiments.stage2_crosslingual import retrieval_p1, train_aligner  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    p.add_argument("--text-model", default="xlm-roberta-large")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmstart-ortho", action="store_true")
    p.add_argument("--controllers", nargs="*", default=available_controllers())
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data = load_crossmodal(args.speech_model, args.text_model, device=device)

    # reuse the Stage-2 aligner by exposing train/test under the names it expects
    class _D:
        dim = data.dim
        X_train, Y_train = data.X_train, data.Y_train
    Xtr, Ytr = data.X_train.to(device), data.Y_train.to(device)
    Xte, Yte = data.X_test.to(device), data.Y_test.to(device)

    print(f"\nSLURP-TN cross-modal alignment  speech→text")
    print(f"  speech {args.speech_model}\n  text   {args.text_model}")
    print(f"train={len(Xtr)}  test={len(Xte)}  dim={data.dim}")
    print("-" * 60)
    print(f"{'method':<22}{'params':>10}{'test P@1':>14}")
    print("-" * 60)
    print(f"{'identity (raw gap)':<22}{'—':>10}{retrieval_p1(Xte, Yte):>14.3f}")
    print(f"{'procrustes (ortho*)':<22}{'—':>10}{retrieval_p1(procrustes(Xtr, Ytr)(Xte), Yte):>14.3f}")
    print(f"{'linear-ls':<22}{'—':>10}{retrieval_p1(linear_ls(Xtr, Ytr)(Xte), Yte):>14.3f}")
    print("-" * 60)

    R_proc = None
    if args.warmstart_ortho:
        Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
        U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
        R_proc = U @ Vt

    rows = []
    for name in args.controllers:
        ws = R_proc if name in ("ortho", "ogated") else None
        ep = 0 if ws is not None else args.epochs
        ctrl = train_aligner(name, _D, ep, args.lr, device, warmstart_R=ws)
        with torch.no_grad():
            acc = retrieval_p1(ctrl(Xte), Yte)
        rows.append((name, ctrl.num_params(), acc))
    for name, params, acc in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<22}{params:>10,}{acc:>14.3f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
