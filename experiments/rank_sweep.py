"""
LoRA-style rank sweep: show the low-rank additive map (I + BA^T) fails to align
independently-trained encoders at EVERY rank, while the orthogonal map (a rank-
CONSTRAINED rotation) wins. Preempts the "you used too-low a rank" objection.

Saves results/rank_sweep.csv and results/rank_sweep.{png,pdf}.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsv.baselines import linear_ls            # noqa: E402
from tasks import load_crosslingual            # noqa: E402

SRC = "intfloat/multilingual-e5-base"
PAIRS = [("deu_Latn", "bert-base-german-cased", "en-de"),
         ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "en-ru"),
         ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
         ("por_Latn", "neuralmind/bert-base-portuguese-cased", "en-pt")]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]


def p1(mapped, tgt):
    a, b = F.normalize(mapped, dim=-1), F.normalize(tgt, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


def main():
    rows, curves = [], {}
    for tgt, enc, label in PAIRS:
        d = load_crosslingual("eng_Latn", tgt, SRC, "cpu", encoder_tgt=enc)
        Xtr, Ytr, Xte, Yte = d.X_train, d.Y_train, d.X_test, d.Y_test
        Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
        U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
        ortho = p1(Xte @ (U @ Vt), Yte)
        lin = p1(linear_ls(Xtr, Ytr)(Xte), Yte)
        W = torch.linalg.lstsq(Xtr, Ytr - Xtr).solution
        Uw, Sw, Vtw = torch.linalg.svd(W, full_matrices=False)
        lr = []
        for r in RANKS:
            Wr = (Uw[:, :r] * Sw[:r]) @ Vtw[:r]
            acc = p1(Xte + Xte @ Wr, Yte)
            lr.append(acc)
            rows.append({"pair": label, "rank": r, "lowrank_p1": round(acc, 4),
                         "ortho_p1": round(ortho, 4), "linear_ls_p1": round(lin, 4)})
        curves[label] = (lr, ortho, lin)
        print(f"{label}: ortho={ortho:.3f} linear-ls={lin:.3f} lowrank r8={lr[3]:.3f} r256={lr[8]:.3f}")

    with open(ROOT / "results" / "rank_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colors = plt.cm.viridis([0.1, 0.4, 0.6, 0.82])
    lang_handles = []
    top_ortho = -1
    for (label, (lr, ortho, lin)), c in zip(curves.items(), colors):
        ax.plot(RANKS, lr, "-o", color=c, ms=4)
        ax.axhline(ortho, ls="--", color=c, alpha=0.85, lw=1.5)
        lang_handles.append(Line2D([0], [0], color=c, lw=2.4, label=label))
        top_ortho = max(top_ortho, ortho)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"rank $r$ of the LoRA-style map $I+BA^\top$")
    ax.set_ylabel("retrieval P@1")
    ax.set_title("LoRA-style maps never reach the orthogonal prior, at any rank")
    ax.set_ylim(-0.03, 1.04)
    # boxed legend 1: map type (ours vs LoRA), top right
    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.7, ls="--", label="ortho (ours), $Rh$"),
        Line2D([0], [0], color="0.3", lw=1.7, ls="-", marker="o", ms=4,
               label="LoRA-style, $I+BA^\\top$"),
    ]
    leg1 = ax.legend(handles=style_handles, fontsize=8.5, loc="upper right",
                     framealpha=0.95)
    ax.add_artist(leg1)
    # boxed legend 2: colour -> language, mid right
    ax.legend(handles=lang_handles, fontsize=8.5, loc="center right",
              title="language pair (colour)", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "results" / f"rank_sweep.{ext}", dpi=150)
    print("saved results/rank_sweep.{csv,png,pdf}")


if __name__ == "__main__":
    main()
