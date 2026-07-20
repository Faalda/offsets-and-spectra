"""
Data efficiency curve: P@1 vs number of alignment pairs (10 → 997).
Shows that ortho and CCA are stable at small n while linear-ls and LoRA collapse.
Runs on cached en→de and en→ru embeddings — no downloads needed.
Saves results/data_efficiency.json and images/data_efficiency.pdf.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual  # noqa: E402

torch.manual_seed(0)

N_VALUES = [10, 25, 50, 100, 200, 500, 997]

PAIRS = [
    ("deu_Latn", "bert-base-german-cased",            "en-de"),
    ("rus_Cyrl", "DeepPavlov/rubert-base-cased",      "en-ru"),
    ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
]


# ── alignment helpers ──────────────────────────────────────────────────────────

def p1(M, Y):
    a, b = F.normalize(M.float(), dim=-1), F.normalize(Y.float(), dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


def run_ortho(Xtr, Ytr, Xte, Yte):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return p1(Xte @ (U @ Vt), Yte)


def run_cca(Xtr, Ytr, Xte, Yte, lam=0.1):
    n = Xtr.shape[0]; mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]; I = torch.eye(d)
    Cxx = Xc.T @ Xc / n + lam * I
    Cyy = Yc.T @ Yc / n + lam * I
    Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C)
        return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy)
    U, _, Vt = torch.linalg.svd(Mx @ Cxy @ My)
    return p1((Xte - mx) @ (Mx @ U), (Yte - my) @ (My @ Vt.T))


def run_linear_ls(Xtr, Ytr, Xte, Yte):
    W, _, _, _ = torch.linalg.lstsq(Xtr, Ytr)
    return p1(Xte @ W, Yte)


def run_lowrank(Xtr, Ytr, Xte, Yte, r=8, steps=400):
    d = Xtr.shape[1]
    A = torch.zeros(d, r, requires_grad=True)
    B = torch.zeros(r, d, requires_grad=True)
    torch.nn.init.xavier_uniform_(A.data)
    torch.nn.init.xavier_uniform_(B.data)
    opt = torch.optim.Adam([A, B], lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        M_upd = A @ B
        loss = (1 - F.cosine_similarity(Xtr + Xtr @ M_upd, Ytr, dim=-1)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        M_upd = A @ B
        return p1(Xte + Xte @ M_upd, Yte)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for tgt_lang, enc_tgt, lab in PAIRS:
        print(f"\n── {lab} ──")
        try:
            d = load_crosslingual(
                "eng_Latn", tgt_lang,
                "intfloat/multilingual-e5-base", "cpu",
                encoder_tgt=enc_tgt,
            )
        except Exception as ex:
            print(f"  FAILED to load: {ex}")
            continue

        Xte, Yte = d.X_test, d.Y_test
        X_all, Y_all = d.X_train, d.Y_train
        n_max = X_all.shape[0]
        print(f"  train pool: {n_max}  test: {Xte.shape[0]}")

        print(f"  {'n':>5} {'ortho':>8} {'cca':>8} {'linear-ls':>10} {'lowrank':>9}")
        pair_results = {}
        rng = torch.Generator(); rng.manual_seed(42)

        for n in N_VALUES:
            if n > n_max:
                continue
            idx = torch.randperm(n_max, generator=rng)[:n]
            Xtr, Ytr = X_all[idx], Y_all[idx]

            scores = {
                "ortho":     run_ortho(Xtr, Ytr, Xte, Yte),
                "cca":       run_cca(Xtr, Ytr, Xte, Yte),
                "linear_ls": run_linear_ls(Xtr, Ytr, Xte, Yte),
                "lowrank":   run_lowrank(Xtr, Ytr, Xte, Yte),
            }
            pair_results[n] = scores
            print(f"  {n:>5} {scores['ortho']:>8.3f} {scores['cca']:>8.3f} "
                  f"{scores['linear_ls']:>10.3f} {scores['lowrank']:>9.3f}")

        all_results[lab] = pair_results

    out = ROOT / "results" / "data_efficiency.json"
    json.dump(all_results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 3.5),
                                 sharey=True)
        if len(all_results) == 1:
            axes = [axes]

        colours = {"ortho": "#2166ac", "cca": "#d6604d",
                   "linear_ls": "#4dac26", "lowrank": "#7b3294"}
        labels  = {"ortho": "ortho (Procrustes)", "cca": "cca (whitened)",
                   "linear_ls": "linear-ls (unreg.)", "lowrank": "LoRA (r=8)"}

        for ax, (lang, rows) in zip(axes, all_results.items()):
            ns = sorted(int(k) for k in rows)
            for key in ("cca", "ortho", "linear_ls", "lowrank"):
                ys = [rows[n][key] for n in ns if n in rows]
                xs = [n for n in ns if n in rows]
                ax.plot(xs, ys, marker="o", label=labels[key],
                        color=colours[key], linewidth=1.8, markersize=4)
            ax.set_xscale("log")
            ax.set_xlabel("# alignment pairs (log scale)")
            ax.set_title(lang)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("retrieval P@1")
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
        fig.suptitle("Data efficiency: P@1 vs number of alignment pairs", y=1.02)
        plt.tight_layout()

        img_dir = ROOT / "images"
        img_dir.mkdir(exist_ok=True)
        out_pdf = img_dir / "data_efficiency.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"Plot saved to {out_pdf}")
    except Exception as ex:
        print(f"  Plot skipped: {ex}")


if __name__ == "__main__":
    main()
