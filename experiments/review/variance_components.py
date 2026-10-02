"""
Clustered inference and variance components for the offset / whitening effects.

POST HOC, DESCRIPTIVE (not pre-registered): answers the pseudo-replication
criticism (gaps share sources and targets) using the crossed 6-source x 17-target
table in results/review/source_families.json (mean-preserving projection).

Responses per gap:  offset cost = centred - probe P@1;  whitening gain = CCA - centred P@1.

(1) Crossed two-way random-effects decomposition (source, target, residual =
    source x target interaction + noise; the design has one observation per cell so
    the two are not separable).  Balanced ANOVA estimators:
        s2_S = (MS_S - MS_E)/T,  s2_T = (MS_T - MS_E)/S,  s2_E = MS_E,  negatives set to 0.
    Uncertainty: pigeonhole bootstrap (resample sources and targets independently).
(2) Trigger tests with the SOURCE held fixed (source fixed effects: demean within source):
      mean dominance of the TARGET (varies across targets)  vs offset cost
      spectral mismatch A (varies across targets)            vs whitening gain
    Spearman on demeaned values; pigeonhole-bootstrap 95% interval.
(3) Source-level exact permutation test of Spearman(source mean dominance, median offset
    cost) over all 6! assignments (one-sided).

Usage: python experiments/review/variance_components.py
Writes: results/review/variance_components.json
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "review"
NATIVE_MD = {"mpnet": 0.084, "gte-qwen2": 0.233, "bert-unc": 0.578,
             "modernbert-large": 0.866, "e5-base": 0.741, "e5-large": 0.734}
B = 4000
rng = np.random.default_rng(0)


def load():
    d = json.load(open(R / "source_families.json"))
    srcs = [k for k in d if not k.startswith("_")]
    tgts = sorted(set.intersection(*[set(d[s]["targets"]) for s in srcs]))
    def M(f): return np.array([[f(d[s]["targets"][t]) for t in tgts] for s in srcs])
    return srcs, tgts, {
        "offset_cost": M(lambda g: g["centred"] - g["probe"]),
        "whiten_gain": M(lambda g: g["cca"] - g["centred"]),
        "centred_p1": M(lambda g: g["centred"]),
        "A": M(lambda g: g["A"]),
        "md_y": M(lambda g: g["mean_dom_y"]),
    }


def components(Y):
    S, T = Y.shape
    gm, rm, cm = Y.mean(), Y.mean(1), Y.mean(0)
    ms_s = T * ((rm - gm) ** 2).sum() / (S - 1)
    ms_t = S * ((cm - gm) ** 2).sum() / (T - 1)
    ms_e = ((Y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((S - 1) * (T - 1))
    v_s, v_t, v_e = max((ms_s - ms_e) / T, 0.0), max((ms_t - ms_e) / S, 0.0), ms_e
    tot = v_s + v_t + v_e
    return np.array([v_s / tot, v_t / tot, v_e / tot]), (v_s, v_t, v_e)


def pigeon(Y, stat, n=B):
    """Pigeonhole bootstrap: resample rows (sources) and columns (targets) independently."""
    S, T = Y.shape
    out = []
    for _ in range(n):
        r = rng.integers(0, S, S); c = rng.integers(0, T, T)
        out.append(stat(Y[np.ix_(r, c)], r, c))
    return np.array(out)


def within_rho(X, Y):
    """Spearman of source-demeaned X and Y over all cells."""
    Xd, Yd = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    return float(spearmanr(Xd.ravel(), Yd.ravel())[0])


def main():
    srcs, tgts, D = load()
    S, T = len(srcs), len(tgts)
    print(f"design: {S} sources x {T} targets = {S*T} gaps")
    out = {"sources": srcs, "n_targets": T, "components": {}, "within_source": {}, "notes":
           "post hoc, descriptive; interaction and noise are not separable (one obs per cell)"}
    print("\n(1) variance components (share of total variance): source / target / residual")
    for k in ["offset_cost", "whiten_gain", "centred_p1"]:
        sh, raw = components(D[k])
        boots = np.array([components(D[k][np.ix_(r, c)])[0]
                          for r, c in ((rng.integers(0, S, S), rng.integers(0, T, T)) for _ in range(B))])
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        out["components"][k] = {"share": sh.tolist(), "ci95_lo": lo.tolist(), "ci95_hi": hi.tolist(),
                                "grand_mean": float(D[k].mean()), "sd_total": float(D[k].std())}
        print(f"  {k:<12} mean={D[k].mean():+.4f}  source {sh[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]  "
              f"target {sh[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]  resid {sh[2]:.2f} [{lo[2]:.2f},{hi[2]:.2f}]")

    print("\n(2) triggers with the source held fixed (Spearman of source-demeaned values)")
    for name, X, Yk in [("target mean dominance -> offset cost", D["md_y"], D["offset_cost"]),
                        ("spectral mismatch A    -> whitening gain", D["A"], D["whiten_gain"])]:
        rho = within_rho(X, Yk)
        bs = np.array([within_rho(X[np.ix_(r, c)], Yk[np.ix_(r, c)])
                       for r, c in ((rng.integers(0, S, S), rng.integers(0, T, T)) for _ in range(B))])
        bs = bs[~np.isnan(bs)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        per_src = [float(spearmanr(X[i], Yk[i])[0]) for i in range(S)]
        out["within_source"][name] = {"rho": rho, "ci95": [float(lo), float(hi)], "per_source_rho": dict(zip(srcs, per_src)),
                                      "n_sources_positive": int(sum(r > 0 for r in per_src))}
        print(f"  {name}: rho={rho:+.3f}  95% [{lo:+.3f},{hi:+.3f}]  per-source: "
              + " ".join(f"{s}:{r:+.2f}" for s, r in zip(srcs, per_src)))

    print("\n(3) source-level exact permutation test, Spearman(source mean dominance, median offset cost)")
    md = np.array([NATIVE_MD[s] for s in srcs]); oc = np.median(D["offset_cost"], axis=1)
    obs = spearmanr(md, oc)[0]
    perms = [spearmanr(md[list(p)], oc)[0] for p in itertools.permutations(range(S))]
    p_one = float(np.mean(np.array(perms) >= obs - 1e-12))
    out["source_level_permutation"] = {"rho": float(obs), "p_one_sided": p_one, "n_perms": len(perms)}
    print(f"  rho={obs:.3f}  exact one-sided p={p_one:.3f} over {len(perms)} assignments")
    (R / "variance_components.json").write_text(json.dumps(out, indent=1))
    print("\nsaved", R / "variance_components.json")


if __name__ == "__main__":
    main()
