"""
Phase-A canonical alignment runner.

Regenerates every cross-lingual alignment number with bootstrap CIs and SAVES
each result under results/align/<pair>/<src_enc>__<tgt_enc>/ as:
    metrics.json   — P@1 + 95% bootstrap CI for every method
    R.pt           — the fitted orthogonal (Procrustes) rotation matrix

Methods reported per pair:
    identity        raw, no map
    cca             whitened rotation (Table tab:align, best column)
    ortho           pure rotation  f(x) = x R         (canonical; paper numbers)
    ortho+shift     rotation+transl f(x) = (x-mu_x)R + mu_y
    linear-ls       unconstrained least-squares linear map
    lowrank         rank-r (r=LOWRANK_R=8) additive (LoRA-style) closed-form MSE
                    NOTE: near-zero P@1 is the EXPECTED result; a rank-8 additive
                    delta cannot represent a full 768-d rotation.  The rank sweep
                    (rank_sweep.py) confirms this holds at every rank.
    dsv             additive shift only

All maps are deterministic closed-form; uncertainty is bootstrap over test queries.
Run:  nadi/.venv/bin/python3 experiments/run_align_all.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsv.baselines import linear_ls            # noqa: E402
from tasks import load_crosslingual            # noqa: E402

import os
SRC_ENC = os.environ.get("SRC_ENC", "intfloat/multilingual-e5-base")
ALIGN_SUBDIR = os.environ.get("ALIGN_SUBDIR", "align")
DEVICE = "cuda:0"
PAIRS = [
    # original four
    ("deu_Latn", "bert-base-german-cased", "en-de"),
    ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "en-ru"),
    ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
    ("zho_Hans", "bert-base-chinese", "en-zh"),
    # expansion: monolingual target BERTs (all hidden dim 768)
    ("fra_Latn", "camembert-base", "en-fr"),
    ("spa_Latn", "dccuchile/bert-base-spanish-wwm-cased", "en-es"),
    ("ita_Latn", "dbmdz/bert-base-italian-cased", "en-it"),
    ("por_Latn", "neuralmind/bert-base-portuguese-cased", "en-pt"),
    ("nld_Latn", "GroNLP/bert-base-dutch-cased", "en-nl"),
    ("tur_Latn", "dbmdz/bert-base-turkish-cased", "en-tr"),
    ("fin_Latn", "TurkuNLP/bert-base-finnish-cased-v1", "en-fi"),
    ("pes_Arab", "HooshvareLab/bert-base-parsbert-uncased", "en-fa"),
    # second expansion: non-Latin / underrepresented families
    ("jpn_Jpan", "cl-tohoku/bert-base-japanese-v3", "en-ja"),
    ("kor_Hang", "klue/bert-base", "en-ko"),
    ("pol_Latn", "dkleczek/bert-base-polish-cased-v1", "en-pl"),
    ("swe_Latn", "KB/bert-base-swedish-cased", "en-sv"),
    ("ind_Latn", "cahya/bert-base-indonesian-1.5G", "en-id"),
]
N_BOOT, SEED, LOWRANK_R = 2000, 0, 8


def fit_R(Xtr, Ytr):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    return U @ Vt


def cca_map(Xtr, Ytr, lam=0.1):
    """Whitened rotation (CCA): projects both spaces into aligned CCA components.
    Returns a closure f(x) that maps source x to CCA space (matched against target CCA space).
    Evaluated as p1(f(Xte), g(Yte)) where g=(Yte-my)@Wy."""
    n = Xtr.shape[0]
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]
    I = torch.eye(d, device=Xtr.device)
    Cxx = Xc.T @ Xc / n + lam * I
    Cyy = Yc.T @ Yc / n + lam * I
    Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C)
        return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy)
    U, _, Vt = torch.linalg.svd(Mx @ Cxy @ My, full_matrices=False)
    Wx = Mx @ U    # (d, d)
    Wy = My @ Vt.T # (d, d)
    return (lambda x: (x - mx) @ Wx), (lambda y: (y - my) @ Wy)


def lowrank_map(Xtr, Ytr, r):
    # closed-form rank-r additive update B A^T minimizing ||X + (X)A B^T - Y||
    # approximate via SVD of residual (Y - X) regressed on X (LoRA-style, rank r)
    W = torch.linalg.lstsq(Xtr, Ytr - Xtr).solution      # d x d least-squares delta
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    return lambda x: x + x @ Wr


def correct(mapped, target):
    a, b = F.normalize(mapped, dim=-1), F.normalize(target, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).cpu().numpy().astype(np.float64)


def ci(c, rng):
    n = len(c)
    m = np.array([c[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(c.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    summary = []
    for tgt, enc_t, label in PAIRS:
      try:
        d = load_crosslingual("eng_Latn", tgt, SRC_ENC, DEVICE, encoder_tgt=enc_t)
        Xtr, Ytr, Xte, Yte = d.X_train, d.Y_train, d.X_test, d.Y_test
        R = fit_R(Xtr, Ytr)
        mu_x, mu_y = Xtr.mean(0), Ytr.mean(0)
        fx_cca, fy_cca = cca_map(Xtr, Ytr)
        maps = {
            "identity":    lambda x: x,
            "cca":         None,           # handled separately (dual-space projection)
            "ortho":       lambda x: x @ R,
            "ortho+shift": lambda x: (x - mu_x) @ R + mu_y,
            "linear-ls":   linear_ls(Xtr, Ytr),
            "lowrank":     lowrank_map(Xtr, Ytr, LOWRANK_R),
            "dsv":         lambda x: x + (mu_y - mu_x),
        }
        metrics = {"pair": label, "src_encoder": SRC_ENC, "tgt_encoder": enc_t,
                   "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                   "dim": int(Xtr.shape[1]), "methods": {}}
        for name, f in maps.items():
            if name == "cca":
                # CCA uses dual projection: compare fx_cca(Xte) vs fy_cca(Yte)
                mean, lo, hi = ci(correct(fx_cca(Xte), fy_cca(Yte)), rng)
            else:
                mean, lo, hi = ci(correct(f(Xte), Yte), rng)
            metrics["methods"][name] = {"p1": mean, "ci95": [lo, hi]}
        outdir = ROOT / "results" / ALIGN_SUBDIR / label / \
            f"{SRC_ENC.replace('/', '_')}__{enc_t.replace('/', '_')}"
        outdir.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(outdir / "metrics.json", "w"), indent=2)
        torch.save(R, outdir / "R.pt")
        summary.append(metrics)
        o = metrics["methods"]
        print(f"{label}: cca {o['cca']['p1']:.3f} | ortho {o['ortho']['p1']:.3f} "
              f"| linear-ls {o['linear-ls']['p1']:.3f} | lowrank(r={LOWRANK_R}) {o['lowrank']['p1']:.4f}"
              f"  -> {outdir.name}")
      except Exception as e:
        print(f"{label}: SKIPPED ({type(e).__name__}: {str(e)[:80]})")
    (ROOT / "results" / ALIGN_SUBDIR).mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(ROOT / "results" / ALIGN_SUBDIR / "summary.json", "w"), indent=2)
    print(f"\nSaved {len(summary)} pairs to results/align/  (+ summary.json)")


if __name__ == "__main__":
    main()
