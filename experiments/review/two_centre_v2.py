"""
Stress test B, BATCH 2: two-centre AND nested-orbital transformations.

Why a second batch: batch 1 (two_centre.py, results/review/two_centre.json) used
rotation strengths b in {0.25, 0.5, 1.0}; every method, identity included, scored
P@1 = 1.000 at every level, so the transformation never disturbed retrieval and
the batch says nothing either way.  The levels here were chosen by looking at
IDENTITY retrieval only (calibration: b=2 -> 0.999, b=3 -> 0.21, b=4 -> 0.00), not
at any aligner's result.  Batch 1 is kept and reported as an uninformative null.
This is a revised, post-hoc batch, not a pre-registered one.

Setting as in two_centre.py: centred top-128 PCs of e5 English (FLORES), groups =
G quantile bins of the first principal coordinate, c_g the group means, R = expm(b*S)
with S a random skew-symmetric matrix of unit spectral norm.

Conditions (Y as a function of the source z in group g):
  distinct : y = c_g + R_g (z - c_g)               (two-centre; centres fixed)
  nested   : y = R_slow c_g + R_fast (z - c_g)      (Sun-Earth-Moon style: the group
             CENTRES move by one rotation, the offsets WITHIN a group by another
             and faster one).  R_slow = expm(b S1), R_fast = expm(3 b S2), shared
             by all groups.  Equivalent to a group-dependent translation on top
             of one rotation.
  global   : y = R z                                (control; recoverable)
Levels b in {0, 2.5, 3.5, 5}; G in {2, 4}; 10 seeds; MLP on the first 3 seeds.

PRE-DECLARED CRITERIA (frozen before any batch-2 result was looked at; identical to
batch 1):
  S1 construction valid  : oracle P@1 >= 0.99.
  S2 statistics matched  : mean A(X,Y) <= 0.10 AND mean shift <= 0.05.
  S3 blind spot          : b>0 with S1, S2 and max(centred Procrustes, CCA) < 0.8.
  S4 diagnostic detects  : D > max D over the b=0 seeds (S3 cells only).
  S5 local map recovers  : piecewise Procrustes (true grouping) beats max(centred,
                           CCA) by >= 0.10 (S3 cells only).
  CONTROL C              : in 'global', max(centred, CCA) >= 0.95 at every b.
All reported whatever the outcome.

Usage: python experiments/review/two_centre_v2.py [device]
Writes: results/review/two_centre_v2.json
"""
from __future__ import annotations
import json, sys
import numpy as np
import torch
import stress_common as S
import common as C
import hub_common as H
from two_centre import rand_rot

BETAS = [0.0, 2.5, 3.5, 5.0]
GROUPS = [2, 4]
CONDS = ["distinct", "nested", "global"]
N_SEEDS, MLP_SEEDS = 10, 3


def build(Ztr, G, cond, b, seed):
    d = Ztr.shape[1]
    gen = torch.Generator().manual_seed(1000 + seed)
    if cond == "global":
        R = rand_rot(d, b, gen)
        return (lambda z: z @ R), (lambda x: x, lambda y: y @ R.T)
    assign = S.make_assign("pc1", Ztr, G)
    g_tr = assign(Ztr)
    cs = [Ztr[g_tr == g].mean(0) for g in range(G)]
    if cond == "distinct":
        Rs = [rand_rot(d, b, gen) for _ in range(G)]

        def fwd(z):
            gs, out = assign(z), torch.empty_like(z)
            for g in range(G):
                s = (gs == g).nonzero().squeeze(1)
                if s.numel():
                    out[s] = cs[g] + (z[s] - cs[g]) @ Rs[g]
            return out
    else:  # nested
        R1, R2 = rand_rot(d, b, gen), rand_rot(d, 3 * b, gen)

        def fwd(z):
            gs, out = assign(z), torch.empty_like(z)
            for g in range(G):
                s = (gs == g).nonzero().squeeze(1)
                if s.numel():
                    out[s] = cs[g] @ R1 + (z[s] - cs[g]) @ R2
            return out
    return fwd, (fwd, lambda y: y)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    Ztr, Zte = S.load_base()
    h = Ztr.shape[0] // 2
    a_null = C.anisotropy(Ztr[:h], Ztr[h:2 * h])
    print(f"base {tuple(Ztr.shape)}  A(X half, X half) = {a_null:.3f}", flush=True)
    rows = []
    for cond in CONDS:
        for G in ([GROUPS[0]] if cond == "global" else GROUPS):
            for b in BETAS:
                for seed in range(N_SEEDS):
                    fwd, oracle_map = build(Ztr, G, cond, b, seed)
                    Ytr_all, Yte = fwd(Ztr), fwd(Zte)
                    oracle = C.evalmap(oracle_map, Zte, Yte)
                    g_ = torch.Generator().manual_seed(seed)
                    idx = torch.randperm(Ztr.shape[0], generator=g_)[: int(0.8 * Ztr.shape[0])]
                    Xtr, Ytr = Ztr[idx], Ytr_all[idx]
                    r = {"G": G, "cond": cond, "b": b, "seed": seed, "oracle": oracle}
                    r["identity"] = C.evalmap(C.identity_map(Xtr, Ytr), Zte, Yte)
                    r["centred"] = C.evalmap(S.global_procrustes(Xtr, Ytr), Zte, Yte)
                    ccav, _, _ = H.cca_val_select(Xtr, Ytr)
                    r["cca"] = C.evalmap(ccav, Zte, Yte)
                    asg = S.make_assign("pc1", Xtr, G)
                    r["pw_true"] = C.evalmap(S.pw_map(Xtr, Ytr, asg, G), Zte, Yte)
                    r["A"] = C.anisotropy(Xtr[:len(Xtr) // 2], Ytr[len(Xtr) // 2:])
                    r["mean_shift"] = S.mean_shift(Xtr, Ytr)
                    r["cov_gap"] = S.cov_gap(Xtr, Ytr)
                    r["D"] = S.diagnostic_D(Xtr, Ytr, seed=seed)
                    if seed < MLP_SEEDS:
                        m, _, _ = C.mlp_tuned(Xtr, Ytr, loss="infonce", device=device, seed=seed)
                        r["mlp"] = C.evalmap(m, Zte, Yte)
                    rows.append(r)
                    print(f"{cond:<8} G={G} b={b:<4} s{seed} id={r['identity']:.3f} cent={r['centred']:.3f} "
                          f"cca={r['cca']:.3f} pw={r['pw_true']:.3f} mlp={r.get('mlp', float('nan')):.3f} "
                          f"orc={oracle:.3f} | A={r['A']:.3f} dmu={r['mean_shift']:.3f} D={r['D']:+.4f}",
                          flush=True)

    def mean(key, **f):
        v = [r[key] for r in rows if key in r and all(r[k] == x for k, x in f.items())]
        return float(np.mean(v)) if v else float("nan")

    keys = ["identity", "centred", "cca", "mlp", "pw_true", "oracle", "A", "mean_shift", "cov_gap", "D"]
    per = {}
    for cond in ["distinct", "nested"]:
        for G in GROUPS:
            null_D = max(r["D"] for r in rows if r["cond"] == cond and r["G"] == G and r["b"] == 0.0)
            for b in BETAS:
                L = {k: mean(k, G=G, cond=cond, b=b) for k in keys}
                lin = max(L["centred"], L["cca"])
                L["S1"] = bool(L["oracle"] >= 0.99)
                L["S2"] = bool(L["A"] <= 0.10 and L["mean_shift"] <= 0.05)
                L["S3"] = bool(b > 0 and L["S1"] and L["S2"] and lin < 0.8)
                L["S4"] = bool(L["D"] > null_D) if L["S3"] else None
                L["S5"] = bool(L["pw_true"] - lin >= 0.10) if L["S3"] else None
                L["null_D"] = null_D
                per[f"{cond}_G{G}_b{b}"] = L
    ctrl = {str(b): {k: mean(k, cond="global", b=b) for k in keys} for b in BETAS}
    ctrl_ok = all(max(v["centred"], v["cca"]) >= 0.95 for v in ctrl.values())
    out = {"rows": rows, "_analysis": {
        "A_null_halves": a_null, "cells": per, "global_control": ctrl,
        "control_C_pass": bool(ctrl_ok), "any_S3": any(v["S3"] for v in per.values()),
        "base": f"{S.SRC} top-{S.K} PCs", "n_seeds": N_SEEDS}}
    (S.ROOT / "results" / "review" / "two_centre_v2.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"\nA(X halves)={a_null:.3f}")
    print(f"{'cell':<20}{'id':>6}{'cent':>7}{'cca':>7}{'mlp':>7}{'pw':>7}{'orc':>6}{'A':>7}{'dmu':>7}{'D':>8}  S1 S2 S3 S4 S5")
    for k, L in per.items():
        print(f"{k:<20}{L['identity']:>6.3f}{L['centred']:>7.3f}{L['cca']:>7.3f}{L['mlp']:>7.3f}"
              f"{L['pw_true']:>7.3f}{L['oracle']:>6.3f}{L['A']:>7.3f}{L['mean_shift']:>7.3f}{L['D']:>+8.4f}  "
              f"{int(L['S1'])}  {int(L['S2'])}  {int(L['S3'])}  {L['S4']}  {L['S5']}")
    print("global control max(centred,cca):", {b: round(max(v['centred'], v['cca']), 3) for b, v in ctrl.items()},
          "pass:", ctrl_ok)
    print("saved two_centre_v2.json; any S3:", out["_analysis"]["any_S3"])


if __name__ == "__main__":
    main()
