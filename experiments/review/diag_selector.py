"""
Prospective NEW batch (Exp 1 of notes.txt): a *diagnostic-driven* 3-branch selector
that uses spectral mismatch A to choose between centred Procrustes and CCA.

    f(X,Y) = identity            if ID_train >= 0.5        (already aligned)
           = centred Procrustes   if A < tau_A             (spectra match: cheaper map suffices)
           = CCA (val-tuned lam)  otherwise                (spectral mismatch: whiten)

DISCIPLINE (notes.txt):
  - tau_A is chosen ON DEVELOPMENT ONLY, to minimise dev mean regret, then FROZEN.
  - Held-out batches (1,2,3: 53 gaps) are evaluated with the frozen tau_A and
    NO retuning. This is a new prospective batch, explicitly distinct from the
    three pre-registered batches (whose tau_A was not frozen before embedding).

Candidate pool of the selector = {identity, centred Procrustes, CCA(val-tuned)}.
Regret is measured against the best AVAILABLE candidate in this pool (identity
unavailable when dims differ, per the paper).  We also report regret against
the FULL candidate set (adds MLP) so the comparison is faithful to Table 5.

Inputs (all precomputed, no embeddings/GPU needed):
  dev:    results/review/selection.json   (id_train, A, test{identity, rotation+centre, cca-tuned})
  held:   results/review/heldout.json, coco_gaps.json, laya_gaps.json
          (diag{ID_train,A}, p1{identity, rotation+centre, cca-val})

Usage:  python experiments/review/diag_selector.py
Writes: results/review/diag_selector.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "review"
ID_T, TIE = 0.5, 0.005


def dev_rows():
    """Valid development instances from selection.json."""
    d = json.loads((R / "selection.json").read_text())
    out = []
    for r in d["rows"]:
        if r["gap"].startswith("xm/libri-whisper"):
            continue
        t = r["test"]
        idt = r["id_train"]
        same = t["identity"] >= 0  # identity available iff dims equal (selection.py sets -1 otherwise)
        pool = {}
        if same:
            pool["identity"] = t["identity"]
        pool["centred"] = t["rotation+centre"]
        pool["cca"] = t["cca-tuned"]
        # full oracle includes mlp-nce (Table 5 honesty)
        full = dict(pool); full["mlp-nce"] = t.get("mlp-nce", -1.0)
        out.append({"gap": r["gap"], "regime": r["regime"], "id_train": idt, "A": r["A"],
                    "pool": pool, "full": full, "same_dim": same})
    return out


def held_rows():
    out = []
    for f in ("heldout.json", "coco_gaps.json", "laya_gaps.json"):
        d = json.loads((R / f).read_text())
        for k, v in d.items():
            if k.startswith("slurp/whisper"):
                continue
            t = v["p1"]; dg = v["diag"]
            idt = dg["ID_train"]
            same = t.get("identity", -1.0) >= 0
            pool = {}
            if same:
                pool["identity"] = t["identity"]
            pool["centred"] = t["rotation+centre"]
            pool["cca"] = t["cca-val"]
            full = dict(pool); full["mlp-nce"] = t.get("mlp-nce", -1.0)
            out.append({"gap": k, "regime": v.get("regime", "?"), "id_train": idt, "A": dg["A"],
                        "pool": pool, "full": full, "same_dim": same})
    return out


def pick(row, tau_A):
    """3-branch diagnostic selector."""
    if row["same_dim"] and row["id_train"] >= ID_T:
        return "identity"
    if row["A"] < tau_A:
        return "centred"
    return "cca"


def pick_paper(row):
    """Paper's Table-5 rule: identity-or-CCA (no centred-Procrustes branch)."""
    if row["same_dim"] and row["id_train"] >= ID_T:
        return "identity"
    return "cca"


def regret(rows, pickfn, vs="pool"):
    """Mean/max regret and accuracy over rows. vs='pool' (selector's own candidates)
    or 'full' (adds MLP, matching Table 5's oracle)."""
    regs, correct = [], []
    for r in rows:
        c = pickfn(r)
        if c not in r["pool"]:      # e.g. identity unavailable -> fall back to cca
            c = "cca"
        best = max((v for v in r[vs].values() if v >= 0), default=0.0)
        regs.append(best - r["pool"][c])
        correct.append(best - r["pool"][c] <= TIE)
    return {"acc": float(np.mean(correct)), "mean": float(np.mean(regs)),
            "max": float(np.max(regs))}


def sweep_tau(rows):
    """Choose tau_A on DEV to minimise mean regret (pool oracle)."""
    cands = sorted({r["A"] for r in rows} | {a + 1e-9 for a in {r["A"] for r in rows}})
    # candidate thresholds = midpoints between consecutive sorted distinct A values
    As = sorted({r["A"] for r in rows})
    ths = [(As[i] + As[i + 1]) / 2 for i in range(len(As) - 1)]
    best_tau, best_mean = None, 1e9
    for tau in ths:
        m = regret(rows, lambda r: pick(r, tau))["mean"]
        if m < best_mean - 1e-12:   # strictly better; ties -> larger tau (more centred-Procrustes)
            best_mean, best_tau = m, tau
        elif m < best_mean + 1e-12 and tau > best_tau:
            best_tau = tau
    return best_tau, best_mean


def fraction_branch_changes(rows, tau_A):
    """Fraction of cases where the spectral branch changes the decision vs the
    paper's identity-or-CCA rule, split by whether it helped/hurt (pool regret)."""
    n, help_, hurt, neutral = 0, 0, 0, 0
    for r in rows:
        c3 = pick(r, tau_A); cp = pick_paper(r)
        if c3 != cp:
            n += 1
            best = max(v for v in r["pool"].values())
            d = (best - r["pool"][c3]) - (best - r["pool"][cp])  # new_regret - old_regret
            if d < -1e-9: help_ += 1
            elif d > 1e-9: hurt += 1
            else: neutral += 1
    return {"n_changed": n, "n_total": len(rows), "helped": help_, "hurt": hurt, "neutral": neutral}


def by_regime(rows, pickfn, tau_A=None):
    out = {}
    for reg in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == reg]
        fn = pickfn if pickfn else (lambda r: pick(r, tau_A))
        out[reg] = {"n": len(sub), **regret(sub, fn)}
    return out


def main():
    dev = dev_rows()
    held = held_rows()
    tau_A, dev_mean = sweep_tau(dev)

    selectors = {
        "3-branch (ID / A<tau -> centred / CCA)": lambda r: pick(r, tau_A),
        "paper: ID-or-CCA": pick_paper,
        "always-CCA": lambda r: "cca",
        "always-centred-Procrustes": lambda r: "centred",
        "always-identity": lambda r: "identity" if r["same_dim"] else "cca",
    }

    summary = {"tau_A_frozen": tau_A,
               "tau_A_chosen_on": f"development ({len(dev)} valid instances, mean regret)",
               "ID_threshold": ID_T, "candidate_pool": ["identity", "centred Procrustes", "CCA(val-tuned)"]}

    summary["development"] = {}
    for nm, fn in selectors.items():
        summary["development"][nm] = {"pool_oracle": regret(dev, fn, "pool"), "full_oracle": regret(dev, fn, "full")}
    summary["development"]["spectral_branch_effect"] = fraction_branch_changes(dev, tau_A)

    summary["held_out"] = {"n": len(held), "tau_A_used": tau_A, "note": "frozen from dev; no retuning"}
    for nm, fn in selectors.items():
        summary["held_out"][nm] = {"pool_oracle": regret(held, fn, "pool"), "full_oracle": regret(held, fn, "full")}
    summary["held_out"]["spectral_branch_effect"] = fraction_branch_changes(held, tau_A)
    summary["held_out"]["by_regime_3branch"] = by_regime(held, None, tau_A)
    summary["held_out"]["by_regime_paper"] = by_regime(held, pick_paper)

    # per-gap detail for held-out
    detail = []
    for r in held:
        c3 = pick(r, tau_A); cp = pick_paper(r)
        best = max(v for v in r["pool"].values())
        detail.append({"gap": r["gap"], "regime": r["regime"], "ID_train": r["id_train"], "A": round(r["A"], 4),
                       "centred": round(r["pool"]["centred"], 4), "cca": round(r["pool"]["cca"], 4),
                       "3branch_pick": c3, "paper_pick": cp,
                       "3branch_regret": round(best - r["pool"][c3], 4),
                       "paper_regret": round(best - r["pool"][cp], 4),
                       "branch_changed_decision": c3 != cp})

    (R / "diag_selector.json").write_text(json.dumps(
        {"summary": summary, "held_out_detail": detail}, indent=1))

    print(f"FROZEN tau_A = {tau_A:.4f}  (chosen on dev, mean regret {dev_mean:.4f})\n")
    for split in ("development", "held_out"):
        s = summary[split]
        print(f"=== {split} (n={s.get('n','?')}) ===")
        for nm, fn in selectors.items():
            m = s[nm]
            print(f"  {nm:<46} pool: acc={m['pool_oracle']['acc']:.2f} "
                  f"mean={m['pool_oracle']['mean']:.4f} max={m['pool_oracle']['max']:.4f} | "
                  f"full: mean={m['full_oracle']['mean']:.4f}")
        b = s["spectral_branch_effect"]
        print(f"  spectral branch changed decision in {b['n_changed']}/{b['n_total']} "
              f"(helped {b['helped']}, hurt {b['hurt']}, neutral {b['neutral']})\n")
    print("by regime (3-branch, held-out):")
    for reg, m in summary["held_out"]["by_regime_3branch"].items():
        print(f"  {reg:<14} n={m['n']} acc={m['acc']:.2f} mean={m['mean']:.4f} max={m['max']:.4f}")
    print("\nsaved", R / "diag_selector.json")


if __name__ == "__main__":
    main()
