"""
Exp 3 (notes.txt): controlled OFFSET x SPECTRAL-MISMATCH factorial.

The only purely CAUSAL test of the paper's decomposition.  Rather than observe
that mean dominance correlates with the centring gain and spectral mismatch
with the whitening gain (Section 5, benchmark-level, shared families), we
CONSTRUCT synthetic gaps from one real frozen embedding and manipulate the two
factors independently:

  Z            = real source embeddings, centered (mean 0, cov Sigma).
  offset o     -> source mean magnitude.  X = Z + o*u  (mean = o*u, cov = Sigma).
                  mean_dom(X) = o^2 / (trace(Sigma) + o^2).  o in {0,med,high}.
  spectral s    -> target covariance distortion.  Y = Z M_s  (mean 0).
                  M_s = V diag(lambda_i^alpha) V^T  (V,lambda: eigendecomp of Sigma).
                  A(X,Y) grows with alpha; A=0 when alpha=0 (M_s=I).
                  alpha in {0, 0.5, 1.0}  (none, med, high).

Both factors are ORTHOGONAL by construction: the offset changes only the mean of
X (covariance fixed); the spectral factor changes only the covariance of Y
(mean fixed at 0).  The true x->y map is affine: x -> (x - o*u) M_s, so CCA is
exact for every (o,s) (Prop. 1) and centred Procrustes is exact only when M_s is
orthogonal (s=0).

Per cell we fit on a TRAIN split and evaluate on a held-out TEST split:
  probe     rotation applied WITHOUT centring (means carried)   -- offset cost
  centred   centred Procrustes (the offset correction)
  cca       CCA(val-tuned lambda) (the whitening correction)
and record
  g_centre  = centred - probe      (the centring gain)
  g_whiten  = cca     - centred    (the whitening gain)
  mean_dom_X, A(X,Y), hub skewness (probe / centred / cca).

PRE-DECLARED SELECTIVITY CRITERION (frozen before any result was looked at):
  The decomposition is causally selective iff BOTH hold across the 9 cells:
    (C1) centring responds to the OFFSET, not the spectrum:
         Spearman(offset_level, g_centre) > +0.4  AND
         |Spearman(spectral_level, g_centre)| < 0.4
    (C2) whitening responds to the SPECTRUM, not the offset:
         Spearman(spectral_level, g_whiten) > +0.4  AND
         |Spearman(offset_level, g_whiten)| < 0.4
  We also report a two-way layout (mean g_centre, g_whiten per factor level) and
  the interaction: does one factor change the OTHER's response?  An interaction
  would qualify, not nullify, selectivity.

Why this matters: a pass is the first causal (not correlational) evidence that
the two corrections dissociate; a fail says the observational correlations in
Section 5 do not survive controlled manipulation.  Both are reportable.

Usage:  python experiments/review/factorial.py [e5|mpnet|bert-unc] [seeds]
Writes: results/review/factorial.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from decomposition import mean_dom  # noqa: E402
from source_families import ld  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
R = ROOT / "results" / "review"

# offset levels target source mean dominance {0, ~0.4, ~0.85}
OFFSET_LEVELS = [0.0, 0.4, 0.85]
# spectral levels: exponent alpha on the eigenvalue scaling {none, med, high}
SPECTRAL_LEVELS = [0.0, 0.5, 1.0]
OFFSET_TAGS = ["o0", "o1", "o2"]
SPECTRAL_TAGS = ["s0", "s1", "s2"]


def build_base(src_tag="intfloat_multilingual-e5-base", tgt="de",
               tgt_enc="bert-base-german-cased"):
    """Real source/target embeddings of one gap.  Returns centered Zx, Zy with a
    REAL (imperfect) correspondence, the source mean direction, the target mean
    (kept fixed), and the target covariance eigendecomp for spectrum distortion.

    Using a real gap (rather than Z=Y) supplies the estimation noise and mild
    nonlinearity that make centring/whitening cost something to recover; the two
    factors are then layered on top orthogonally."""
    X = ld(src_tag, "eng_Latn", "dev").float()
    Y = ld(tgt_enc, tgt, "dev").float()
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]
    mx, my = X.mean(0), Y.mean(0)
    Zx, Zy = X - mx, Y - my
    # target covariance eigendecomp for constructing M_s
    Sigma_y = Zy.T @ Zy / n
    w, V = torch.linalg.eigh(Sigma_y)
    w = w.clamp(min=1e-6)
    u = mx / (mx.norm() + 1e-12)                       # unit source mean direction
    return Zx, Zy, u, my, (w, V), n


def make_M(w, V, alpha):
    """Spectrum-distorting symmetric map on the target: M = V diag(lambda^alpha) V^T.
    alpha=0 -> identity (no distortion); larger alpha -> stronger mismatch A(X,Y)."""
    scale = w ** alpha
    return (V * scale) @ V.T


def construct(Zx, Zy, u, my, wV, o_level, s_level, seed=0):
    """Build one synthetic gap from real centered embeddings:
       X = Zx + o*u        (source mean = o*u, covariance = cov(Zx) fixed)
       Y = Zy M_s + my     (target mean = my FIXED, covariance distorted by M_s)
    The offset factor changes ONLY the source mean; the spectral factor changes
    ONLY the target covariance.  Means of both spaces are present (my kept), so
    the probe sees the real mean-mean hub mechanism."""
    w, V = wV
    g = torch.Generator().manual_seed(seed)
    n, d = Zx.shape
    # offset: scale u so that mean_dom(X) ~= o_level
    trSx = float((Zx ** 2).sum() / n)                   # trace(cov(Zx))
    if o_level > 0:
        o2 = o_level / (1.0 - o_level) * trSx if o_level < 1 else 1e6
        o_mag = float(o2 ** 0.5)
        X = Zx + o_mag * u
    else:
        X = Zx.clone()
    # spectral distortion of target covariance, target mean kept fixed
    M = make_M(w, V, s_level)
    Y = Zy @ M + my
    # train/test split (80/20)
    perm = torch.randperm(n, generator=g)
    ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]
    return X[tr], Y[tr], X[te], Y[te]


def eval_cell(Xtr, Ytr, Xte, Yte):
    rot = C.ortho_map(Xtr, Ytr)
    probe = C.evalmap(rot, Xte, Yte)
    cm = H.centred_map(rot, Xtr, Ytr)
    centred = C.evalmap(cm, Xte, Yte)
    ccav, _, _ = H.cca_val_select(Xtr, Ytr)
    cca = C.evalmap(ccav, Xte, Yte)
    return {
        "probe": probe, "centred": centred, "cca": cca,
        "g_centre": centred - probe, "g_whiten": cca - centred,
        "mean_dom_X": mean_dom(Xtr),
        "A": C.anisotropy(Xtr[:Xtr.shape[0] // 2], Ytr[Xtr.shape[0] // 2:]),  # unpaired
        "hub_probe": H.map_hubness(rot, Xte, Yte, k=10),
        "hub_centred": H.map_hubness(cm, Xte, Yte, k=10),
        "hub_cca": H.map_hubness(ccav, Xte, Yte, k=10),
    }


def main():
    src_tag = sys.argv[1] if len(sys.argv) > 1 else "intfloat_multilingual-e5-base"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    # pick a representative real target gap from the benchmark XL list
    from benchmark import XL  # noqa
    tgt_lang, tgt_enc, tgt_lab = next(x for x in XL if x[2] == "de")
    Zx, Zy, u, my, wV, n = build_base(src_tag, tgt_lang, tgt_enc.replace("/", "_"))
    d = Zx.shape[1]
    rows = []
    cells = {}  # (oi,si) -> list of dicts
    for si, (o, ot) in enumerate(zip(OFFSET_LEVELS, OFFSET_TAGS)):
        for sj, (s, st) in enumerate(zip(SPECTRAL_LEVELS, SPECTRAL_TAGS)):
            cells[(si, sj)] = []
            for seed in range(n_seeds):
                Xtr, Ytr, Xte, Yte = construct(Zx, Zy, u, my, wV, o, s, seed=seed)
                r = eval_cell(Xtr, Ytr, Xte, Yte)
                r.update({"o_level": si, "s_level": sj, "o_tag": ot, "s_tag": st, "seed": seed})
                rows.append(r)
                cells[(si, sj)].append(r)
                print(f"{ot}{st} seed{seed} md={r['mean_dom_X']:.3f} A={r['A']:.3f} "
                      f"probe={r['probe']:.3f} cent={r['centred']:.3f} cca={r['cca']:.3f} "
                      f"g_centre={r['g_centre']:+.4f} g_whiten={r['g_whiten']:+.4f} "
                      f"hub_p/c/cc={r['hub_probe']:.1f}/{r['hub_centred']:.1f}/{r['hub_cca']:.1f}",
                      flush=True)

    # ---- pre-declared selectivity test ----
    o_lvl = np.array([r["o_level"] for r in rows], dtype=float)
    s_lvl = np.array([r["s_level"] for r in rows], dtype=float)
    gc = np.array([r["g_centre"] for r in rows])
    gw = np.array([r["g_whiten"] for r in rows])
    rho_oc, p_oc = spearmanr(o_lvl, gc)
    rho_sc, p_sc = spearmanr(s_lvl, gc)
    rho_ow, p_ow = spearmanr(o_lvl, gw)
    rho_sw, p_sw = spearmanr(s_lvl, gw)
    c1 = (rho_oc > 0.4) and (abs(rho_sc) < 0.4)
    c2 = (rho_sw > 0.4) and (abs(rho_ow) < 0.4)

    # two-way cell means (for interaction)
    cell_means = {}
    for (i, j), rs in cells.items():
        cell_means[f"{OFFSET_TAGS[i]}{SPECTRAL_TAGS[j]}"] = {
            "g_centre": float(np.mean([r["g_centre"] for r in rs])),
            "g_whiten": float(np.mean([r["g_whiten"] for r in rs])),
            "mean_dom_X": float(np.mean([r["mean_dom_X"] for r in rs])),
            "A": float(np.mean([r["A"] for r in rs])),
            "n": len(rs),
        }

    # marginal means per factor level (interaction check)
    marg_o, marg_s = {}, {}
    for i in range(3):
        rs = [r for r in rows if r["o_level"] == i]
        marg_o[OFFSET_TAGS[i]] = {"g_centre": float(np.mean([r["g_centre"] for r in rs])),
                                  "g_whiten": float(np.mean([r["g_whiten"] for r in rs]))}
    for j in range(3):
        rs = [r for r in rows if r["s_level"] == j]
        marg_s[SPECTRAL_TAGS[j]] = {"g_centre": float(np.mean([r["g_centre"] for r in rs])),
                                    "g_whiten": float(np.mean([r["g_whiten"] for r in rs]))}

    analysis = {
        "predeclared_criterion": (
            "C1: rho(offset,g_centre)>0.4 AND |rho(spectral,g_centre)|<0.4; "
            "C2: rho(spectral,g_whiten)>0.4 AND |rho(offset,g_whiten)|<0.4."),
        "selectivity": {
            "rho_offset_gcentre": float(rho_oc), "p_offset_gcentre": float(p_oc),
            "rho_spectral_gcentre": float(rho_sc), "p_spectral_gcentre": float(p_sc),
            "rho_offset_gwhiten": float(rho_ow), "p_offset_gwhiten": float(p_ow),
            "rho_spectral_gwhiten": float(rho_sw), "p_spectral_gwhiten": float(p_sw),
            "C1_pass": bool(c1), "C2_pass": bool(c2),
            "selective": bool(c1 and c2),
        },
        "marginal_means_offset": marg_o,
        "marginal_means_spectral": marg_s,
        "cell_means": cell_means,
        "n_rows": len(rows), "n_seeds": n_seeds, "base": f"{src_tag} -> {tgt_lab}",
    }
    out = {"rows": rows, "_analysis": analysis}
    (R / "factorial.json").write_text(json.dumps(out, indent=1, default=float))

    a = analysis["selectivity"]
    print("\n=== PRE-DECLARED SELECTIVITY ===")
    print(f"  rho(offset, g_centre)  = {a['rho_offset_gcentre']:+.3f} (p={a['p_offset_gcentre']:.3g})")
    print(f"  rho(spectral, g_centre)= {a['rho_spectral_gcentre']:+.3f} (p={a['p_spectral_gcentre']:.3g})")
    print(f"  rho(offset, g_whiten)  = {a['rho_offset_gwhiten']:+.3f} (p={a['p_offset_gwhiten']:.3g})")
    print(f"  rho(spectral, g_whiten)= {a['rho_spectral_gwhiten']:+.3f} (p={a['p_spectral_gwhiten']:.3g})")
    print(f"  C1 (centring<-offset)  : {'PASS' if a['C1_pass'] else 'FAIL'}")
    print(f"  C2 (whiten<-spectrum)  : {'PASS' if a['C2_pass'] else 'FAIL'}")
    print(f"  SELECTIVE             : {'YES' if a['selective'] else 'NO'}")
    print("\n=== marginal means (offset) ===")
    for k, v in marg_o.items():
        print(f"  {k}  g_centre={v['g_centre']:+.4f}  g_whiten={v['g_whiten']:+.4f}")
    print("=== marginal means (spectral) ===")
    for k, v in marg_s.items():
        print(f"  {k}  g_centre={v['g_centre']:+.4f}  g_whiten={v['g_whiten']:+.4f}")
    print("\nsaved", R / "factorial.json")


if __name__ == "__main__":
    main()
