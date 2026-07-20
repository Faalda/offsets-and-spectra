"""
Predictive analysis: does a geometry statistic predict the benefit of whitening?

For every gap we measure (a) an anisotropy statistic and (b) Delta = CCA - ortho,
the gain of whitening over a pure rotation. If anisotropy predicts Delta, the
framework becomes predictive: measure geometry -> choose controller.

Saves results/predictive/{summary.json, delta_vs_aniso.{pdf,png}}.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual            # noqa: E402
torch.manual_seed(0)


def p1(M, Y):
    a, b = F.normalize(M, dim=-1), F.normalize(Y, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()

def ortho(a, b, c, e):
    Xc, Yc = a - a.mean(0), b - b.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return p1(c @ (U @ Vt), e)

def cca(a, b, c, e, lam=0.1):
    n = a.shape[0]; mx, my = a.mean(0), b.mean(0); Xc, Yc = a - mx, b - my
    d = Xc.shape[1]; I = torch.eye(d)
    Cxx = Xc.T @ Xc / n + lam * I; Cyy = Yc.T @ Yc / n + lam * I; Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C); return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy); U, S, Vt = torch.linalg.svd(Mx @ Cxy @ My)
    return p1((c - mx) @ (Mx @ U), (e - my) @ (My @ Vt.T))

def aniso(a, b):
    def spec(Z):
        Zc = Z - Z.mean(0); C = Zc.T @ Zc / Z.shape[0]
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-9); return (ev / ev.sum()).sort(descending=True).values
    return (spec(a) - spec(b)).abs().sum().item()

def logcond(a, b):
    def lc(Z):
        Zc = Z - Z.mean(0); ev = torch.linalg.eigvalsh(Zc.T @ Zc / Z.shape[0]).clamp(min=1e-9)
        return torch.log10(ev.max() / ev.min()).item()
    return 0.5 * (lc(a) + lc(b))


XL = [("deu_Latn","bert-base-german-cased"),("rus_Cyrl","DeepPavlov/rubert-base-cased"),
      ("arb_Arab","CAMeL-Lab/bert-base-arabic-camelbert-da"),("zho_Hans","bert-base-chinese"),
      ("fra_Latn","camembert-base"),("spa_Latn","dccuchile/bert-base-spanish-wwm-cased"),
      ("ita_Latn","dbmdz/bert-base-italian-cased"),("por_Latn","neuralmind/bert-base-portuguese-cased"),
      ("nld_Latn","GroNLP/bert-base-dutch-cased"),("tur_Latn","dbmdz/bert-base-turkish-cased"),
      ("fin_Latn","TurkuNLP/bert-base-finnish-cased-v1"),("pes_Arab","HooshvareLab/bert-base-parsbert-uncased"),
      ("jpn_Jpan","cl-tohoku/bert-base-japanese-v3"),
      ("kor_Hang","klue/bert-base"),
      ("pol_Latn","dkleczek/bert-base-polish-cased-v1"),
      ("swe_Latn","KB/bert-base-swedish-cased"),
      ("ind_Latn","cahya/bert-base-indonesian-1.5G")]
CA = [("bert-base-uncased","roberta-base"),("bert-base-uncased","microsoft/deberta-base"),
      ("roberta-base","microsoft/deberta-base"),("bert-base-uncased","google/electra-base-discriminator")]


def record(a, b, c, e, name, regime):
    o, cc = ortho(a, b, c, e), cca(a, b, c, e)
    return dict(gap=name, regime=regime, aniso=aniso(a, b), logcond=logcond(a, b),
                ortho=o, cca=cc, delta=cc - o)


def main():
    rows = []
    # controlled rotation (isotropic)
    d = load_crosslingual("eng_Latn", "deu_Latn", "intfloat/multilingual-e5-base", "cpu")
    X, Xt = d.X_train, d.X_test; Q, _ = torch.linalg.qr(torch.randn(X.shape[1], X.shape[1]))
    rows.append(record(X, X @ Q, Xt, Xt @ Q, "controlled-rot", "isotropic"))
    for tgt, enc in XL:
        d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
        rows.append(record(d.X_train, d.Y_train, d.X_test, d.Y_test, f"xl-{tgt[:3]}", "cross-lingual"))
    for ea, eb in CA:
        d = load_crosslingual("eng_Latn", "eng_Latn", ea, "cpu", encoder_tgt=eb)
        rows.append(record(d.X_train, d.Y_train, d.X_test, d.Y_test,
                           f"xa-{ea.split('/')[-1][:4]}/{eb.split('/')[-1][:4]}", "cross-arch"))
    try:
        from tasks.crossmodal import load_crossmodal
        cm = load_crossmodal("openai/whisper-large-v3", "xlm-roberta-large", device="cpu")
        rows.append(record(cm.X_train, cm.Y_train, cm.X_test, cm.Y_test, "xmodal", "nonlinear"))
    except Exception as ex:
        print(f"(xmodal skipped: {ex})")

    A = np.array([r["aniso"] for r in rows]); D = np.array([r["delta"] for r in rows])
    L = np.array([r["logcond"] for r in rows])
    # Delta is only meaningful where a linear geometry applies; fit on those.
    lin = [r for r in rows if r["regime"] != "nonlinear"]
    Al = np.array([r["aniso"] for r in lin]); Dl = np.array([r["delta"] for r in lin])
    def pearson(x, y): return float(np.corrcoef(x, y)[0, 1])
    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
    stats = dict(pearson_aniso_all=pearson(A, D), pearson_aniso_lin=pearson(Al, Dl),
                 spearman_aniso_lin=spearman(Al, Dl),
                 pearson_logcond=pearson(L, D))
    out = ROOT / "results" / "predictive"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rows=rows, stats=stats), open(out / "summary.json", "w"), indent=2)
    for r in rows:
        print(f"{r['gap']:<16}{r['regime']:<14}aniso={r['aniso']:.2f} delta={r['delta']:+.3f}")
    print("\ncorr(aniso,delta) linear-only: pearson={pearson_aniso_lin:.3f} "
          "spearman={spearman_aniso_lin:.3f}".format(**stats))
    print("corr(aniso,delta) all: pearson={pearson_aniso_all:.3f} | "
          "logcond pearson={pearson_logcond:.3f}".format(**stats))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = {"isotropic": "#999999", "cross-lingual": "#4c72b0", "cross-arch": "#55a868", "nonlinear": "#8172b3"}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for reg in cmap:
        xs = [r["aniso"] for r in rows if r["regime"] == reg]
        ys = [r["delta"] for r in rows if r["regime"] == reg]
        ax.scatter(xs, ys, c=cmap[reg], label=reg, s=45, edgecolor="k", linewidth=0.4)
    m, bq = np.polyfit(Al, Dl, 1); xs = np.linspace(Al.min(), Al.max(), 50)
    ax.plot(xs, m * xs + bq, "k--", lw=1.2,
            label=f"fit, linear gaps (r={stats['pearson_aniso_lin']:.2f})")
    ax.set_xlabel("anisotropy of the gap"); ax.set_ylabel(r"$\Delta$ = CCA $-$ rotation (whitening benefit)")
    ax.set_title("Whitening helps in proportion to anisotropy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"delta_vs_aniso.{ext}", dpi=150)
    print("saved results/predictive/{summary.json, delta_vs_aniso.pdf}")


if __name__ == "__main__":
    main()
