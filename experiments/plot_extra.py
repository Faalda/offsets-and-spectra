"""
Two reviewer-driven figures for the DSV paper:
 (1) method ladder: avg retrieval P@1 across the alignment ladder, showing that
     geometry-respecting maps (CCA, ortho) beat regularized/unregularized flexible
     linear maps (PLS, linear-ls, ridge) and low-rank/LoRA -- regularization alone
     does not close the gap; structure does.
 (2) rotation invariance: the anisotropy diagnostic is unchanged under random
     orthogonal pre-rotations of the source space (a function of covariance spectra).

Saves paper/fig_method_ladder.pdf and paper/fig_rotation_invariance.pdf.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
sys.path.insert(0, str(ROOT))

# ---- shared palette (matches geometry-spectrum / predictive figures) ----
BLUE, GREEN, ORANGE, RED, PURPLE, GREY = "#4c72b0", "#55a868", "#dd8452", "#c44e52", "#8172b3", "#999999"

# ---- (1) method ladder (avg P@1 over the 12 pairs, from regularized_baselines.py) ----
LADDER = [("CCA", 0.953, GREEN), ("ortho", 0.848, BLUE),
          ("PLS", 0.669, ORANGE), ("linear-ls", 0.464, ORANGE),
          ("ridge", 0.346, ORANGE), ("lowrank\n(LoRA)", 0.003, RED)]


def plot_ladder():
    names = [n for n, _, _ in LADDER]; vals = [v for _, v, _ in LADDER]
    cols = [c for _, _, c in LADDER]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(range(len(names)), vals, color=cols, edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("avg retrieval P@1 (12 language pairs)"); ax.set_ylim(0, 1.05)
    ax.set_title("Structure beats capacity: geometry-respecting maps win")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, BLUE, ORANGE, RED)]
    ax.legend(handles, ["CCA (whiten+rotate)", "ortho (rotate)", "flexible linear (PLS/LS/ridge)",
                        "low-rank / LoRA"], fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(PAPER / "fig_method_ladder.pdf"); fig.savefig(PAPER / "fig_method_ladder.png", dpi=150)
    print("saved fig_method_ladder")


# ---- (2) rotation invariance of the anisotropy diagnostic ----
def anisotropy(a, b):
    def spec(Z):
        Zc = Z - Z.mean(0); C = Zc.T @ Zc / Z.shape[0]
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-9); return (ev / ev.sum()).sort(descending=True).values
    return (spec(a) - spec(b)).abs().sum().item()


def plot_invariance():
    from tasks import load_crosslingual
    torch.manual_seed(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    PAIRS = [("deu_Latn", "bert-base-german-cased", "en-de"),
             ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
             ("por_Latn", "neuralmind/bert-base-portuguese-cased", "en-pt")]
    pcols = {"en-de": BLUE, "en-ar": ORANGE, "en-pt": GREEN}
    for tgt, enc, lab in PAIRS:
        d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
        X, Y = d.X_train, d.Y_train
        vals = [anisotropy(X, Y)]  # original
        for _ in range(10):
            Q, _ = torch.linalg.qr(torch.randn(X.shape[1], X.shape[1]))
            vals.append(anisotropy(X @ Q, Y))
        ax.plot(range(len(vals)), vals, "-o", ms=4, color=pcols[lab], label=f"{lab} (base {vals[0]:.3f})")
    ax.set_xlabel("random orthogonal pre-rotation of source (0 = none)")
    ax.set_ylabel("anisotropy diagnostic")
    ax.set_title("The anisotropy diagnostic is invariant to orthogonal pre-rotation")
    ax.set_ylim(0, None); ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(PAPER / "fig_rotation_invariance.pdf"); fig.savefig(PAPER / "fig_rotation_invariance.png", dpi=150)
    print("saved fig_rotation_invariance")


# ---- (3) retrieval similarity-matrix grid (DSV analog of an attention-map grid) ----
def plot_simgrid(pair=("deu_Latn", "bert-base-german-cased", "en-de"), n=30):
    from tasks import load_crosslingual
    from dsv.baselines import linear_ls          # paper's exact unconstrained linear map
    import torch.nn.functional as F
    tgt, enc, lab = pair
    d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
    Xtr, Ytr = d.X_train, d.Y_train
    Xfull, Yfull = d.X_test, d.Y_test            # full test set (for representative P@1)
    Xte, Yte = d.X_test[:n], d.Y_test[:n]        # subset (for a legible heatmap)
    dd = Xtr.shape[1]
    def full_p1(mapx, mapy):                     # top-1 accuracy over the FULL test set
        a, b = F.normalize(mapx, dim=-1), F.normalize(mapy, dim=-1)
        return (a @ b.T).argmax(1).eq(torch.arange(a.shape[0])).float().mean().item()
    # per-row retrieval distribution P(target j | source i) = softmax over the subset
    def sim(a, b, tau=0.04):
        S = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
        return torch.softmax(S / tau, dim=1)
    mx, my = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mx, Ytr - my
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False); R = U @ Vt
    f_lin = linear_ls(Xtr, Ytr)                 # matches paper's linear-ls column
    I = torch.eye(dd); Cxx = Xc.T@Xc/Xtr.shape[0]+0.1*I; Cyy = Yc.T@Yc/Xtr.shape[0]+0.1*I; Cxy = Xc.T@Yc/Xtr.shape[0]
    def isq(C): w, V = torch.linalg.eigh(C); return V@torch.diag(w.clamp(min=1e-6)**-0.5)@V.T
    Mx, My = isq(Cxx), isq(Cyy); Uc, _, Vtc = torch.linalg.svd(Mx@Cxy@My)
    # (name, subset heatmap, full-test P@1)
    panels = [("identity", sim(Xte, Yte), full_p1(Xfull, Yfull)),
              ("linear-ls", sim(f_lin(Xte), Yte), full_p1(f_lin(Xfull), Yfull)),
              ("ortho", sim(Xte @ R, Yte), full_p1(Xfull @ R, Yfull)),
              ("CCA", sim((Xte - mx) @ (Mx @ Uc), (Yte - my) @ (My @ Vtc.T)),
               full_p1((Xfull - mx) @ (Mx @ Uc), (Yfull - my) @ (My @ Vtc.T)))]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9))
    for ax, (name, S, acc) in zip(axes, panels):
        im = ax.imshow(S.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{name}\nP@1={acc:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("target index", fontsize=7)
    axes[0].set_ylabel("source index", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="retrieval prob.")
    fig.suptitle(f"Retrieval similarity ({lab}): full-test P@1 in titles; "
                 f"heatmaps show a {n}-example subset (softmax over candidates)", fontsize=9)
    fig.savefig(PAPER / "fig_simgrid.pdf"); fig.savefig(PAPER / "fig_simgrid.png", dpi=150)
    print("saved fig_simgrid")


if __name__ == "__main__":
    plot_ladder()
    plot_invariance()
    plot_simgrid()
