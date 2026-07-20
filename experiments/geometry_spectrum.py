"""
Geometry spectrum: measure the geometry of each representation gap, then show
which controller geometry wins. The organizing experiment of the paper.

Regimes:
  isotropic    controlled rotation on one encoder (gap is a pure rotation)
  anisotropic  cross-lingual cross-encoder gaps (gap is a whitened rotation)
  nonlinear    cross-modal speech<->text gaps (no linear/rotational structure)

For each task we report retrieval P@1 of:
  ortho  (rotation only)   |  cca (whiten+rotate)  |  mlp (nonlinear)
and an anisotropy statistic of the gap. Saves results/geometry/summary.json
and results/geometry/spectrum.{pdf,png}.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual            # noqa: E402

torch.manual_seed(0)


def p1(M, Y):
    a, b = F.normalize(M, dim=-1), F.normalize(Y, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


def ortho(Xtr, Ytr, Xte, Yte):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return p1(Xte @ (U @ Vt), Yte)


def cca(Xtr, Ytr, Xte, Yte, lam=0.1):
    n = Xtr.shape[0]; mx, my = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]; I = torch.eye(d)
    Cxx = Xc.T @ Xc / n + lam * I; Cyy = Yc.T @ Yc / n + lam * I; Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C); return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy)
    U, S, Vt = torch.linalg.svd(Mx @ Cxy @ My)
    return p1((Xte - mx) @ (Mx @ U), (Yte - my) @ (My @ Vt.T))


def mlp(Xtr, Ytr, Xte, Yte, h=512, steps=400):
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, d))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = (1 - F.cosine_similarity(net(Xtr), Ytr, dim=-1)).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return p1(net(Xte), Yte)


def anisotropy(Xtr, Ytr):
    """How much does whitening change the alignment geometry = covariance-spectrum
    mismatch between source and target. 0 = identical (pure rotation suffices),
    larger = more anisotropic (whitening helps)."""
    def spec(Z):
        Zc = Z - Z.mean(0); C = Zc.T @ Zc / Z.shape[0]
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-9); ev = ev / ev.sum()
        return ev.sort(descending=True).values
    sx, sy = spec(Xtr), spec(Ytr)
    return (sx - sy).abs().sum().item()       # L1 distance of normalized eigen-spectra


def main():
    rows = []
    # 1) isotropic: controlled rotation on a single encoder
    d = load_crosslingual("eng_Latn", "deu_Latn", "intfloat/multilingual-e5-base", "cpu")
    X, Xt = d.X_train, d.X_test
    Q, _ = torch.linalg.qr(torch.randn(X.shape[1], X.shape[1]))
    rows.append(dict(task="controlled-rot", regime="isotropic",
                     aniso=anisotropy(X, X @ Q),
                     ortho=ortho(X, X @ Q, Xt, Xt @ Q), cca=cca(X, X @ Q, Xt, Xt @ Q),
                     mlp=mlp(X, X @ Q, Xt, Xt @ Q)))
    # 2) anisotropic: cross-lingual cross-encoder
    XL = [("deu_Latn", "bert-base-german-cased", "xling-de"),
          ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "xling-ru"),
          ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "xling-ar")]
    for tgt, enc, lab in XL:
        d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
        a, b, c, e = d.X_train, d.Y_train, d.X_test, d.Y_test
        rows.append(dict(task=lab, regime="anisotropic", aniso=anisotropy(a, b),
                         ortho=ortho(a, b, c, e), cca=cca(a, b, c, e), mlp=mlp(a, b, c, e)))
    # 3) nonlinear: cross-modal speech<->text
    try:
        from tasks.crossmodal import load_crossmodal
        for sm, lab in [("openai/whisper-large-v3", "xmodal-whisper")]:
            cm = load_crossmodal(sm, "xlm-roberta-large", device="cpu")
            a, b, c, e = cm.X_train, cm.Y_train, cm.X_test, cm.Y_test
            rows.append(dict(task=lab, regime="nonlinear", aniso=anisotropy(a, b),
                             ortho=ortho(a, b, c, e), cca=cca(a, b, c, e), mlp=mlp(a, b, c, e)))
    except Exception as ex:
        print(f"(cross-modal skipped: {type(ex).__name__}: {str(ex)[:60]})")

    out = ROOT / "results" / "geometry"; out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "summary.json", "w"), indent=2)
    print(f"{'task':<16}{'regime':<13}{'aniso':>7}{'ortho':>8}{'cca':>8}{'mlp':>8}")
    for r in rows:
        print(f"{r['task']:<16}{r['regime']:<13}{r['aniso']:>7.2f}"
              f"{r['ortho']:>8.3f}{r['cca']:>8.3f}{r['mlp']:>8.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r["task"] for r in rows]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w, [r["ortho"] for r in rows], w, label="ortho (rotate)", color="#4c72b0")
    ax.bar(x,     [r["cca"]   for r in rows], w, label="CCA (whiten+rotate)", color="#55a868")
    ax.bar(x + w, [r["mlp"]   for r in rows], w, label="MLP (nonlinear)", color="#8172b3")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("retrieval P@1"); ax.set_ylim(0, 1.05)
    ax.set_title("Which geometry wins depends on the gap's geometry")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for ext in ("pdf", "png"):
        fig.tight_layout(); fig.savefig(out / f"spectrum.{ext}", dpi=150)
    print("saved results/geometry/summary.json + spectrum.{pdf,png}")


if __name__ == "__main__":
    main()
