"""
Numerical checks for the revised theory (A1-A4 in paper/review_todo.md).

(1) LoRA residual class: singular spectrum of (R* - I) for fitted Procrustes R*.
    Prop (revised): I + BA^T represents R exactly iff r >= rank(R - I); otherwise
    min_{B,A} ||R - I - BA^T||_F^2 = sum_{i>r} sigma_i(R - I)^2 (Eckart-Young).
    We report rank and the fraction of ||R-I||_F^2 NOT capturable at r = 8/64/256.

(2) Exactness of whitening for ANY invertible linear gap (no isotropy):
    Y = X M  =>  whitened spaces are related by an orthogonal K.  Verified on
    real anisotropic X (e5 embeddings) with a random ill-conditioned M.

(3) Spectral lower bound on the error of ANY isometric map (no isotropy, no
    linear-gap model): with both spaces scaled to unit total variance,
        min_R E||y - xR||^2  >=  ||lam(S_Y) - lam(S_X)||_2^2 / (||S_Y||^.5 + ||S_X||^.5)^2
    Compared with the achieved Procrustes MSE on real pairs.

(4) Data efficiency: is there ANY training size at which ortho beats CCA?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402
from benchmark import XL  # noqa: E402

torch.set_default_dtype(torch.float64)


def unit_var(Z, mu):
    Zc = Z - mu
    return Zc / Zc.pow(2).sum(1).mean().sqrt()


def cov(Z):
    Zc = Z - Z.mean(0)
    return Zc.T @ Zc / Z.shape[0]


def main():
    out = {"lora": {}, "bound": {}}
    for tgt, enc, lab in XL[:12]:
        d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
        X, Y, Xt, Yt = (t.double() for t in (d.X_train, d.Y_train, d.X_test, d.Y_test))
        # (1)
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        U, _, Vt = torch.linalg.svd(Xc.T @ Yc); R = U @ Vt
        s = torch.linalg.svdvals(R - torch.eye(768))
        tot = (s ** 2).sum()
        out["lora"][lab] = {"rank_R_minus_I(sv>1e-6)": int((s > 1e-6).sum()),
                            "det_R": float(torch.linalg.det(R)),
                            **{f"unexplained_frac_r{r}": float((s[r:] ** 2).sum() / tot) for r in (8, 64, 256, 512)},
                            "mean_sv": float(s.mean())}
        # (3) bound, on TEST data, both spaces centred + unit total variance, Procrustes fitted on train
        mx, my = X.mean(0), Y.mean(0)
        Xu, Yu = unit_var(X, mx), unit_var(Y, my)
        U, _, Vt = torch.linalg.svd(Xu.T @ Yu); Ru = U @ Vt
        Xtu, Ytu = unit_var(Xt, Xt.mean(0)), unit_var(Yt, Yt.mean(0))
        mse_rot = float((Ytu - Xtu @ Ru).pow(2).sum(1).mean())
        U2, _, Vt2 = torch.linalg.svd(Xtu.T @ Ytu); mse_rot_oracle = float((Ytu - Xtu @ (U2 @ Vt2)).pow(2).sum(1).mean())
        SX, SY = cov(Xtu), cov(Ytu)
        lx, ly = torch.linalg.eigvalsh(SX).flip(0), torch.linalg.eigvalsh(SY).flip(0)
        bound = float((ly - lx).pow(2).sum() / (ly[0].sqrt() + lx[0].sqrt()) ** 2)
        out["bound"][lab] = {"A": float((lx / lx.sum() - ly / ly.sum()).abs().sum()),
                             "bound": bound, "mse_rot_test_oracleR": mse_rot_oracle,
                             "mse_rot_test_trainR": mse_rot}
        print(lab, out["lora"][lab]["rank_R_minus_I(sv>1e-6)"],
              round(out["lora"][lab]["unexplained_frac_r64"], 3), out["bound"][lab], flush=True)
    # (2) exactness on real anisotropic X
    d = load_crosslingual("eng_Latn", "deu_Latn", "intfloat/multilingual-e5-base", "cpu",
                          encoder_tgt="bert-base-german-cased")
    X = d.X_train.double(); Xt = d.X_test.double()
    g = torch.Generator().manual_seed(0)
    Q1, _ = torch.linalg.qr(torch.randn(768, 768, generator=g)); Q2, _ = torch.linalg.qr(torch.randn(768, 768, generator=g))
    M = Q1 @ torch.diag(torch.logspace(-1, 1, 768)) @ Q2     # condition number 100
    Y, Yt = X @ M, Xt @ M
    ex = {}
    for lam in (0.0, 1e-6, 1e-3):
        ex[f"cca_lam{lam}"] = C.evalmap(C.cca_map(X, Y, lam), Xt, Yt)
    ex["ortho"] = C.evalmap(C.ortho_map(X, Y), Xt, Yt)
    ex["A"] = C.anisotropy(X, Y)
    ex["self_aniso_x"] = C.self_anisotropy(X)
    # whitened-space orthogonality residual of the population K
    SX = cov(X); SY = cov(Y)
    isq = lambda S: (lambda w, V: (V * w.clamp(min=1e-12).rsqrt()) @ V.T)(*torch.linalg.eigh(S))
    sq = lambda S: (lambda w, V: (V * w.clamp(min=0).sqrt()) @ V.T)(*torch.linalg.eigh(S))
    K = sq(SX) @ M @ isq(SY)
    ex["||K^T K - I||_F"] = float((K.T @ K - torch.eye(768)).norm())
    out["exactness"] = ex
    print("exactness", ex)
    # (4)
    de = json.loads((ROOT / "results" / "data_efficiency.json").read_text())
    viol = [(p, n, v["ortho"], v["cca"]) for p, dd in de.items() for n, v in dd.items() if v["ortho"] > v["cca"]]
    out["data_eff_ortho_beats_cca"] = viol
    print("ortho>cca at any n:", viol)
    p = ROOT / "results" / "review" / "theory_checks.json"
    p.write_text(json.dumps(out, indent=1)); print("saved", p)


if __name__ == "__main__":
    main()
