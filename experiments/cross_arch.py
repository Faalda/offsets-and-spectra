"""
Cross-architecture alignment (same language, different encoders).

Answers the reviewer question "is the whitened-rotation geometry only a
cross-lingual phenomenon?" We embed the SAME English sentences (FLORES) with two
independently trained, architecturally distinct encoders and align them. If
CCA > ortho >> linear/LoRA again, the geometry is a property of independently
trained representation spaces, not of languages.

Saves results/crossarch/summary.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsv.baselines import linear_ls            # noqa: E402
from tasks import load_crosslingual            # noqa: E402

N_BOOT = 2000
# English-only, all hidden dim 768; gap is purely architectural
PAIRS = [
    ("bert-base-uncased", "roberta-base", "BERT-RoBERTa"),
    ("bert-base-uncased", "microsoft/deberta-base", "BERT-DeBERTa"),
    ("roberta-base", "microsoft/deberta-base", "RoBERTa-DeBERTa"),
    ("bert-base-uncased", "google/electra-base-discriminator", "BERT-ELECTRA"),
]


def p1c(M, Y):
    a, b = F.normalize(M, dim=-1), F.normalize(Y, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).cpu().numpy().astype(np.float64)


def ci(c, rng):
    n = len(c); m = np.array([c[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(c.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ortho_map(Xtr, Ytr):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return lambda x: x @ (U @ Vt)


def cca_map(Xtr, Ytr, lam=0.1):
    n = Xtr.shape[0]; mx, my = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]; I = torch.eye(d)
    Cxx = Xc.T @ Xc / n + lam * I; Cyy = Yc.T @ Yc / n + lam * I; Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C); return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy); U, S, Vt = torch.linalg.svd(Mx @ Cxy @ My)
    Wx, Wy = Mx @ U, My @ Vt.T
    return lambda x: (x - mx) @ Wx, lambda y: (y - my) @ Wy


def anisotropy(X, Y):
    def spec(Z):
        Zc = Z - Z.mean(0); C = Zc.T @ Zc / Z.shape[0]
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-9); ev = ev / ev.sum()
        return ev.sort(descending=True).values
    return (spec(X) - spec(Y)).abs().sum().item()


def main():
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'pair':<18}{'aniso':>7}{'ortho':>8}{'cca':>8}{'lin-ls':>8}")
    for ea, eb, lab in PAIRS:
        try:
            d = load_crosslingual("eng_Latn", "eng_Latn", ea, "cpu", encoder_tgt=eb)
            Xtr, Ytr, Xte, Yte = d.X_train, d.Y_train, d.X_test, d.Y_test
            o = ci(p1c(ortho_map(Xtr, Ytr)(Xte), Yte), rng)
            fx, fy = cca_map(Xtr, Ytr)
            c = ci(p1c(fx(Xte), fy(Yte)), rng)
            l = ci(p1c(linear_ls(Xtr, Ytr)(Xte), Yte), rng)
            a = anisotropy(Xtr, Ytr)
            rows.append(dict(pair=lab, src=ea, tgt=eb, aniso=a,
                             ortho=o, cca=c, linear_ls=l))
            print(f"{lab:<18}{a:>7.2f}{o[0]:>8.3f}{c[0]:>8.3f}{l[0]:>8.3f}")
        except Exception as ex:
            print(f"{lab:<18} SKIPPED ({type(ex).__name__}: {str(ex)[:60]})")
    out = ROOT / "results" / "crossarch"; out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "summary.json", "w"), indent=2)
    if rows:
        print(f"\navg ortho={np.mean([r['ortho'][0] for r in rows]):.3f} "
              f"cca={np.mean([r['cca'][0] for r in rows]):.3f} "
              f"lin-ls={np.mean([r['linear_ls'][0] for r in rows]):.3f}")
    print("saved results/crossarch/summary.json")


if __name__ == "__main__":
    main()
