"""
Extend expanded_survey.json with new cross-lingual pairs for both e5-base and LaBSE
source encoders, then recompute final_stats.json.

Run after adding new entries to run_align_all.py and predictive.py XL lists.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual

NEW_PAIRS = [
    ("jpn_Jpan", "cl-tohoku/bert-base-japanese-v3"),
    ("kor_Hang", "klue/bert-base"),
    ("pol_Latn", "dkleczek/bert-base-polish-cased-v1"),
    ("swe_Latn", "KB/bert-base-swedish-cased"),
    ("ind_Latn", "cahya/bert-base-indonesian-1.5G"),
]
SRC_ENCODERS = [
    ("e5", "intfloat/multilingual-e5-base"),
    ("labse", "sentence-transformers/LaBSE"),
]


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

def record(a, b, c, e, name, regime):
    o, cc = ortho(a, b, c, e), cca(a, b, c, e)
    return dict(gap=name, regime=regime, aniso=aniso(a, b), logcond=logcond(a, b),
                ortho=o, cca=cc, delta=cc - o)

def pearson_ci(x, y, n_boot=5000, seed=42):
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        rs.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    return r, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    survey_path = ROOT / "results" / "predictive" / "expanded_survey.json"
    stats_path = ROOT / "results" / "predictive" / "final_stats.json"

    existing = json.loads(survey_path.read_text())
    rows = existing["rows"]
    existing_gaps = {r["gap"] for r in rows}
    print(f"Existing rows: {len(rows)}")

    new_rows = []
    for prefix, src_enc in SRC_ENCODERS:
        for tgt, enc_t in NEW_PAIRS:
            lang_short = tgt[:3].lower()
            gap_name = f"{prefix}/{lang_short}"
            if gap_name in existing_gaps:
                print(f"  SKIP (already present): {gap_name}")
                continue
            print(f"  Computing {gap_name} ({src_enc} → {enc_t}) ...", flush=True)
            try:
                d = load_crosslingual("eng_Latn", tgt, src_enc, "cpu", encoder_tgt=enc_t)
                row = record(d.X_train, d.Y_train, d.X_test, d.Y_test, gap_name, "cross-lingual")
                new_rows.append(row)
                print(f"    aniso={row['aniso']:.3f}  delta={row['delta']:+.3f}")
            except Exception as ex:
                print(f"    FAILED: {ex}")

    if not new_rows:
        print("No new rows added.")
        return

    rows = rows + new_rows
    survey_path.write_text(json.dumps({"rows": rows, "stats": {}}, indent=2))
    print(f"\nSaved {len(rows)} rows to {survey_path}")

    # Recompute final_stats on cross-lingual + isotropic only (exclude cross-arch)
    xl_iso = [r for r in rows if r["regime"] in ("cross-lingual", "isotropic")]
    e5_xl = [r for r in xl_iso if r["gap"].startswith("e5/") or r["gap"].startswith("ctrl-rot")]
    labse_xl = [r for r in xl_iso if r["gap"].startswith("labse/")]

    def compute_stats(subset, label):
        A = np.array([r["aniso"] for r in subset])
        D = np.array([r["delta"] for r in subset])
        r_val, lo, hi = pearson_ci(A, D)
        p_val = float(scipy_stats.pearsonr(A, D)[1])
        sp = float(np.corrcoef(np.argsort(np.argsort(A)), np.argsort(np.argsort(D)))[0, 1])
        print(f"  {label}: n={len(subset)}  r={r_val:.4f}  CI=[{lo:.4f},{hi:.4f}]  p={p_val:.4f}  spearman={sp:.4f}")
        return {"n": len(subset), "r": r_val, "p": p_val, "spearman": sp, "ci_lo": lo, "ci_hi": hi}

    print("\nRecomputed correlations:")
    final_stats = {
        "combined_xl_iso": compute_stats(xl_iso, "combined_xl_iso"),
        "e5_only": compute_stats(e5_xl, "e5_only"),
        "labse_only": compute_stats(labse_xl, "labse_only"),
    }
    stats_path.write_text(json.dumps(final_stats, indent=2))
    print(f"Saved final_stats to {stats_path}")


if __name__ == "__main__":
    main()
