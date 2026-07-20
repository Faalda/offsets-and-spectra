"""
Stage-2 experiment: cross-lingual sentence-embedding alignment (FLORES-200).

Each controller g learns to map source-language embeddings into the target
space; we evaluate cross-lingual retrieval P@1 on held-out pairs and compare
against two analytic reference lines:

    procrustes  — optimal orthogonal map  (what `ortho` should match)
    linear-ls   — optimal linear map      (what `affine`/`matrix` should match)
    identity    — raw e5, no alignment    (how aligned the encoder already is)

Run:
    nadi/.venv/bin/python3 experiments/stage2_crosslingual.py --src eng_Latn --tgt arb_Arab
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import build_controller, available_controllers          # noqa: E402
from dsv.baselines import procrustes, linear_ls                  # noqa: E402
from tasks import load_crosslingual, random_rotation             # noqa: E402


@torch.no_grad()
def retrieval_p1(mapped_src: torch.Tensor, tgt: torch.Tensor) -> float:
    """src→tgt nearest-neighbour P@1 under cosine (pairs are aligned by index)."""
    a = F.normalize(mapped_src, dim=-1)
    b = F.normalize(tgt, dim=-1)
    sims = a @ b.T                                  # (N, N)
    pred = sims.argmax(dim=1)
    gold = torch.arange(a.shape[0], device=a.device)
    return (pred == gold).float().mean().item()


def train_aligner(name, data, epochs, lr, device, warmstart_R=None, **kw):
    ctrl = build_controller(name, dim=data.dim, n_layers=1, **kw).to(device)
    X, Y = data.X_train.to(device), data.Y_train.to(device)
    # Closed-form warm start: Cayley+Adam over SO(d) is poorly conditioned in high
    # dim, so start the orthogonal family at the Procrustes optimum and fine-tune.
    if warmstart_R is not None and hasattr(ctrl, "set_rotation"):
        ctrl.set_rotation(warmstart_R.T.to(device))
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        # maximise cosine to the true pair == MSE on L2-normalised vectors
        loss = (1 - F.cosine_similarity(ctrl(X), Y, dim=-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
        opt.step()
    ctrl.eval()
    return ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="eng_Latn")
    p.add_argument("--tgt", default="arb_Arab")
    p.add_argument("--encoder", default="intfloat/multilingual-e5-base")
    p.add_argument("--encoder-tgt", default=None,
                   help="second encoder for the target side (Stage 2b natural gap); "
                        "must share the source encoder's hidden dim")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rotate-target", action="store_true",
                   help="apply a fixed random rotation to the target space — "
                        "a controlled rigid gap on real embeddings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmstart-ortho", action="store_true",
                   help="initialise ortho/ogated from the closed-form Procrustes "
                        "rotation before gradient fine-tuning")
    p.add_argument("--controllers", nargs="*", default=available_controllers())
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data = load_crosslingual(args.src, args.tgt, args.encoder, device,
                             encoder_tgt=args.encoder_tgt)
    Xte, Yte = data.X_test.to(device), data.Y_test.to(device)
    Xtr, Ytr = data.X_train.to(device), data.Y_train.to(device)

    if args.rotate_target:
        g = torch.Generator().manual_seed(args.seed)
        Q = random_rotation(data.dim, g).to(device)
        Ytr, Yte = Ytr @ Q, Yte @ Q              # same rotation on both splits
        data.Y_train, data.Y_test = Ytr.cpu(), Yte.cpu()   # so train_aligner sees it

    rot = " | +rotation" if args.rotate_target else ""
    enc_line = args.encoder if not args.encoder_tgt else f"{args.encoder} → {args.encoder_tgt}"
    print(f"\nFLORES-200 alignment  {args.src} → {args.tgt}{rot}  | encoder {enc_line}")
    print(f"train pairs={len(Xtr)}  test pairs={len(Xte)}  dim={data.dim}")
    print("-" * 60)
    print(f"{'method':<22}{'params':>10}{'test P@1':>14}")
    print("-" * 60)

    # analytic reference lines (fit on train, eval on test)
    print(f"{'identity (raw e5)':<22}{'—':>10}{retrieval_p1(Xte, Yte):>14.3f}")
    f_proc = procrustes(Xtr, Ytr)
    print(f"{'procrustes (ortho*)':<22}{'—':>10}{retrieval_p1(f_proc(Xte), Yte):>14.3f}")
    f_ls = linear_ls(Xtr, Ytr)
    print(f"{'linear-ls (affine*)':<22}{'—':>10}{retrieval_p1(f_ls(Xte), Yte):>14.3f}")
    print("-" * 60)

    # closed-form Procrustes rotation, for optional warm start of the ortho family
    R_proc = None
    if args.warmstart_ortho:
        Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
        U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
        R_proc = U @ Vt

    rows = []
    for name in args.controllers:
        ws = R_proc if name in ("ortho", "ogated") else None
        # The orthogonal controller's optimum on a single linear layer is the
        # closed-form Procrustes solution; gradient descent over Cayley overfits
        # with few pairs, so when warm-started we report the closed form (0 steps).
        ep = 0 if ws is not None else args.epochs
        ctrl = train_aligner(name, data, ep, args.lr, device, warmstart_R=ws)
        with torch.no_grad():
            acc = retrieval_p1(ctrl(Xte), Yte)
        rows.append((name, ctrl.num_params(), acc))
    for name, params, acc in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<22}{params:>10,}{acc:>14.3f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
