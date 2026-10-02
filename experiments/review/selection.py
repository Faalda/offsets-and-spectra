"""
AI-W3: evaluate the diagnostic as a PROSPECTIVE controller selector.

Candidates: identity, ortho, cca (lam=0.1, fixed), nonlinear (mlp-nce, tuned).
Test P@1 of each candidate per gap comes from results/review/benchmark_*.json.

Selectors (all thresholds fixed a priori, before looking at any test outcome):
  diag        the paper's Table-1 rule with explicit thresholds:
                ID_train >= 0.5              -> identity
                else CCA val-P@1 < 0.5       -> nonlinear   (linear ceiling low)
                else A < A_STAR              -> ortho
                else                         -> cca
              ID_train: identity retrieval on the train pairs (no map fitted)
              A: anisotropy from UNPAIRED halves of the train sets
              CCA val-P@1: CCA fitted on 80% of train, scored on the other 20%
  diag-noA    same without the anisotropy branch (identity / cca / nonlinear)
  always-X    for each candidate X
  val-select  fit every candidate on 80% of train, pick the best on 20% (standard CV)
  oracle      best on test (upper bound)
Reports accuracy (= picks the oracle-best, ties within 0.005 count as correct),
mean and max regret (oracle P@1 - chosen P@1), per regime.

Usage: python experiments/review/selection.py
"""
from __future__ import annotations
import json, sys, ast
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402

A_STAR, ID_T, CEIL_T, TIE = 0.32, 0.5, 0.5, 0.005
CANDS = ["identity", "ortho", "rotation+centre", "cca", "cca-tuned", "mlp-nce"]


def main():
    bench = {}
    for f in (ROOT / "results" / "review").glob("benchmark_*.json"):
        bench.update(json.loads(f.read_text()))
    HD = json.loads((ROOT / "results" / "review" / "hub_dev.json").read_text())
    rows = []
    for group in ("ctrl", "xling_e5", "xling_labse", "xarch", "clip", "xmodal"):
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
            if name not in bench:
                continue
            b = bench[name]; test = {k: b["p1"].get(k, -1.0) for k in CANDS}
            test["rotation+centre"] = HD[name]["p1"]["rotation+centre"]
            same = Xtr.shape[1] == Ytr.shape[1]
            a, bb, va, vb = C.split_val(Xtr, Ytr)
            val = {"identity": C.evalmap(C.identity_map(a, bb), va, vb) if same else -1.0,
                   "ortho": C.evalmap(C.ortho_map(a, bb), va, vb),
                   "rotation+centre": C.evalmap(H.centred_map(C.ortho_map(a, bb), a, bb), va, vb),
                   "cca": C.evalmap(C.cca_map(a, bb), va, vb),
                   "cca-tuned": max(C.evalmap(C.cca_map(a, bb, l), va, vb) for l in C.LAMBDAS_CCA),
                   "mlp-nce": ast.literal_eval(b["hp"]["mlp-nce"])["val_p1"]}
            idtr = C.p1(Xtr, Ytr) if same else 0.0
            A = b["diag"]["aniso_unpaired"]
            if idtr >= ID_T:
                diag = "identity"
            elif val["cca-tuned"] < CEIL_T:
                diag = "mlp-nce"
            elif A < A_STAR:
                diag = "ortho"
            else:
                diag = "cca-tuned"
            diag_noA = "identity" if idtr >= ID_T else ("mlp-nce" if val["cca-tuned"] < CEIL_T else "cca-tuned")
            diag_2 = "identity" if idtr >= ID_T else "cca-tuned"
            diag_2_fixed = "identity" if idtr >= ID_T else "cca"
            picks = {"diag(submitted:A-branch)": diag, "diag(ID,ceiling)": diag_noA, "diag(ID only)": diag_2,
                     "diag(ID only, cca lam=.1)": diag_2_fixed, "val-select": max(val, key=val.get),
                     **{f"always-{k}": k for k in CANDS}}
            best = max(test.values())
            rows.append({"gap": name, "regime": regime, "id_train": idtr, "A": A, "val": val,
                         "test": test, "oracle": max(test, key=test.get), "picks": picks,
                         "regret": {s: best - test[k] if test[k] >= 0 else best for s, k in picks.items()},
                         "correct": {s: bool(best - test[k] <= TIE) for s, k in picks.items()}})
    # aggregate; LibriSpeech's 5 split-seeds count as ONE gap (averaged)
    def agg(rs):
        out = {}
        for s in rows[0]["picks"]:
            reg = np.array([r["regret"][s] for r in rs]); cor = np.array([r["correct"][s] for r in rs])
            out[s] = {"acc": float(cor.mean()), "mean_regret": float(reg.mean()), "max_regret": float(reg.max())}
        return out
    summary = {"n_gaps": len(rows), "all": agg(rows)}
    for reg in sorted({r["regime"] for r in rows}):
        rs = [r for r in rows if r["regime"] == reg]
        summary[reg] = {"n": len(rs), **agg(rs)}
    out = ROOT / "results" / "review" / "selection.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for k, v in summary.items():
        if isinstance(v, dict):
            print(k, v.get("n", ""))
            for s, m in v.items():
                if isinstance(m, dict):
                    print(f"   {s:<18} acc={m['acc']:.2f} regret mean={m['mean_regret']:.3f} max={m['max_regret']:.3f}")
    print("oracle picks:", {r["gap"]: r["oracle"] for r in rows})
    print("saved", out)


if __name__ == "__main__":
    main()
