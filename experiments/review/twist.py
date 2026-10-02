"""
Stress test A (notes.txt): position-dependent SPIRAL TWIST.

Question: can two spaces have (nearly) the same mean and covariance spectrum, so
that the paper's diagnostics see no mismatch, while a nonlinear twist defeats
every global linear aligner?

Construction (deterministic; oracle inverse exists exactly).
  Z      = centred top-128 principal coordinates of e5 English (FLORES).
  r(z)   = ||z||, standardised on FLORES dev: u = (r - mean r) / std r.
  T_a(z) = block-diagonal rotation of the 64 adjacent coordinate planes
           (0,1),(2,3),... by the SAME angle theta = a * u(z) for that point.
  Y      = T_a(Z).   Each point is rotated by an angle set by its own distance
           from the centre (a in radians per standard deviation of the norm).
  T_a preserves ||z||, so u(T_a(z)) = u(z) and the inverse T_a^{-1} (rotate by
  -theta) is exact.  a=0 gives Y = Z.
Note: the notes' theta = alpha*||x|| would be almost constant because e5
embeddings have nearly constant norm and would collapse to a global rotation,
so the angle is standardised.  Adjacent planes have similar variances, which
keeps the covariance spectrum as close to preserved as a twist can.

Levels a in {0, 0.25, 0.5, 1, 2}; 10 seeds (random 80% of FLORES dev for fitting;
FLORES devtest, 1,012 pairs, for evaluation).  MLP (InfoNCE, validation grid) on
the first MLP_SEEDS seeds only.

PRE-DECLARED CRITERIA (frozen before any result was looked at):
  S1 construction valid   : oracle-inverse P@1 >= 0.99 at every level.
  S2 statistics matched   : at a level, mean over seeds of A(X,Y) <= 0.10 AND
                            mean shift ||mu_Y-mu_X||/rms <= 0.05.
                            (Reference: A between two disjoint halves of X.)
  S3 blind spot           : some level satisfies S1 and S2 and has
                            max(centred Procrustes, CCA) P@1 < 0.8 (identity is
                            reported for context).
  S4 diagnostic detects   : at each S3 level, structure diagnostic D (label-free
                            cross-validated piecewise gain, stress_common.diagnostic_D)
                            exceeds the MAXIMUM D over the a=0 seeds.
  S5 local map recovers   : informed piecewise Procrustes (norm bins, G=8) beats
                            max(centred, CCA) by >= 0.10 at each S3 level.
  Each criterion is reported pass/fail whatever the outcome.  If no level meets S2,
  the experiment says nothing about the diagnostics and is reported as such.

Usage: python experiments/review/twist.py [device]
Writes: results/review/twist.json
"""
from __future__ import annotations
import json, sys
import numpy as np
import torch
from stress_common import *  # noqa: F401,F403
import stress_common as S
import common as C
import hub_common as H

LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS, MLP_SEEDS = 10, 3
PLANES = S.K // 2


def norm_stats(Ztr):
    r = Ztr.norm(dim=1)
    return float(r.mean()), float(r.std())


def twist(Z, a, stats, sign=1.0):
    """Rotate every adjacent coordinate plane of each point by sign*a*u(Z)."""
    m, s = stats
    th = sign * a * ((Z.norm(dim=1) - m) / s)
    c, sn = th.cos().unsqueeze(1), th.sin().unsqueeze(1)
    P = Z.view(Z.shape[0], PLANES, 2)
    u, v = P[..., 0], P[..., 1]
    return torch.stack([c * u - sn * v, sn * u + c * v], dim=-1).reshape_as(Z)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    Ztr, Zte = S.load_base()
    stats = norm_stats(Ztr)
    h = Ztr.shape[0] // 2
    a_null = C.anisotropy(Ztr[:h], Ztr[h:2 * h])
    print(f"base dims {tuple(Ztr.shape)}  A(X half, X half) = {a_null:.3f}", flush=True)
    rows = []
    for a in LEVELS:
        Ytr_all, Yte = twist(Ztr, a, stats), twist(Zte, a, stats)
        oracle = C.evalmap((lambda x: x, lambda y: twist(y, a, stats, sign=-1.0)), Zte, Yte)
        for seed in range(N_SEEDS):
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(Ztr.shape[0], generator=g)[: int(0.8 * Ztr.shape[0])]
            Xtr, Ytr = Ztr[idx], Ytr_all[idx]
            r = {"a": a, "seed": seed, "oracle": oracle}
            r["identity"] = C.evalmap(C.identity_map(Xtr, Ytr), Zte, Yte)
            r["centred"] = C.evalmap(S.global_procrustes(Xtr, Ytr), Zte, Yte)
            ccav, _, _ = H.cca_val_select(Xtr, Ytr)
            r["cca"] = C.evalmap(ccav, Zte, Yte)
            for name, kind, G in [("pw_norm8", "norm", 8), ("pw_norm4", "norm", 4),
                                  ("pw_pc1_4", "pc1", 4)]:
                asg = S.make_assign(kind, Xtr, G)
                r[name] = C.evalmap(S.pw_map(Xtr, Ytr, asg, G), Zte, Yte)
            r["A"] = C.anisotropy(Xtr[:len(Xtr) // 2], Ytr[len(Xtr) // 2:])
            r["mean_shift"] = S.mean_shift(Xtr, Ytr)
            r["cov_gap"] = S.cov_gap(Xtr, Ytr)
            r["D"] = S.diagnostic_D(Xtr, Ytr, seed=seed)
            r["hub_centred"] = H.map_hubness(S.global_procrustes(Xtr, Ytr), Zte, Yte, k=10)
            if seed < MLP_SEEDS:
                m, _, _ = C.mlp_tuned(Xtr, Ytr, loss="infonce", device=device, seed=seed)
                r["mlp"] = C.evalmap(m, Zte, Yte)
            rows.append(r)
            print(f"a={a:<4} s{seed} id={r['identity']:.3f} cent={r['centred']:.3f} cca={r['cca']:.3f} "
                  f"pwN8={r['pw_norm8']:.3f} pwP4={r['pw_pc1_4']:.3f} "
                  f"mlp={r.get('mlp', float('nan')):.3f} orc={oracle:.3f} | "
                  f"A={r['A']:.3f} dmu={r['mean_shift']:.3f} covgap={r['cov_gap']:.3f} D={r['D']:+.4f}",
                  flush=True)

    # ---------------- analysis (criteria as pre-declared in the docstring)
    def mean(key, a):
        v = [r[key] for r in rows if r["a"] == a and key in r]
        return float(np.mean(v)) if v else float("nan")

    null_D = max(r["D"] for r in rows if r["a"] == 0.0)
    per_level = {}
    for a in LEVELS:
        lin = max(mean("centred", a), mean("cca", a))
        per_level[str(a)] = {k: mean(k, a) for k in
                             ["identity", "centred", "cca", "mlp", "pw_norm8", "pw_norm4", "pw_pc1_4",
                              "oracle", "A", "mean_shift", "cov_gap", "D", "hub_centred"]}
        L = per_level[str(a)]
        L["S1"] = bool(L["oracle"] >= 0.99)
        L["S2"] = bool(L["A"] <= 0.10 and L["mean_shift"] <= 0.05)
        L["S3"] = bool(a > 0 and L["S1"] and L["S2"] and lin < 0.8)
        L["S4"] = bool(L["D"] > null_D) if L["S3"] else None
        L["S5"] = bool(L["pw_norm8"] - lin >= 0.10) if L["S3"] else None
        L["best_linear"] = lin
    out = {"rows": rows, "_analysis": {
        "A_null_halves": a_null, "D_null_max_a0": null_D, "per_level": per_level,
        "any_S3": any(v["S3"] for v in per_level.values()),
        "S1_all": all(v["S1"] for v in per_level.values()),
        "base": f"{S.SRC} top-{S.K} PCs, FLORES eng dev->devtest", "n_seeds": N_SEEDS}}
    (S.ROOT / "results" / "review" / "twist.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\nA(X halves)={a_null:.3f}  null max D (a=0) = {null_D:+.4f}  S1(all)={out['_analysis']['S1_all']}")
    print(f"{'a':<5}{'id':>6}{'cent':>7}{'cca':>7}{'mlp':>7}{'pwN8':>7}{'pwP4':>7}{'orc':>6}"
          f"{'A':>7}{'dmu':>7}{'covg':>7}{'D':>8}  S1 S2 S3 S4 S5")
    for a in LEVELS:
        L = per_level[str(a)]
        print(f"{a:<5}{L['identity']:>6.3f}{L['centred']:>7.3f}{L['cca']:>7.3f}{L['mlp']:>7.3f}"
              f"{L['pw_norm8']:>7.3f}{L['pw_pc1_4']:>7.3f}{L['oracle']:>6.3f}{L['A']:>7.3f}"
              f"{L['mean_shift']:>7.3f}{L['cov_gap']:>7.3f}{L['D']:>+8.4f}  "
              f"{int(L['S1'])}  {int(L['S2'])}  {int(L['S3'])}  {L['S4']}  {L['S5']}")
    print("saved twist.json; any level satisfying S3 (blind spot):", out["_analysis"]["any_S3"])


if __name__ == "__main__":
    main()
