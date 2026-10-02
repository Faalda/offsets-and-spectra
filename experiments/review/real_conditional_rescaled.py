"""
POST HOC sensitivity check for batch 5 (real_conditional.py).  Not pre-registered.

The declared detector D is a squared-error gain and the image encoders' features have norms of 10-47
against 1.0 for the text embeddings, so D on cross-modal gaps is dominated by scale.  Here each side is
divided by its training-set mean row norm before the identical procedure is run.  The declared result
of real_conditional.py is unchanged; this only tests whether the negative outcome is a scale artefact.

Usage: python experiments/review/real_conditional_rescaled.py
Writes: results/review/real_conditional_rescaled.json
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "review"))
import real_conditional as RC

res = {}
for name, reg, Xtr, Ytr, Xte, Yte in RC.gaps():
    sx, sy = Xtr.float().norm(dim=1).mean(), Ytr.float().norm(dim=1).mean()
    r = RC.run_gap(Xtr / sx, Ytr / sy, Xte / sx, Yte / sy, full_cca=False)
    r["regime"] = reg; res[name] = r
    m = r["selected_minus_cca"]
    print(f"{name:<44} maxD={r['max_D']:+.4f} sel={r['selected']:<13} centred={r['p1']['centred']:.3f} cca128={r['p1']['cca']:.3f} "
          f"sel={r['p1']['selected']:.3f} diff={m['mean']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]", flush=True)
pos = [k for k, v in res.items() if v["selected_minus_cca"]["mean"] >= 0.02 and v["selected_minus_cca"]["ci95"][0] > 0]
summ = {"gaps_with_selected_gain_ge_0.02_and_ci_gt_0": pos, "gaps_with_maxD_gt_0": [k for k, v in res.items() if v["max_D"] > 0], "n": len(res)}
(Path(__file__).resolve().parents[2] / "results/review/real_conditional_rescaled.json").write_text(json.dumps({"gaps": res, "summary": summ}, indent=1, default=float))
print("SUMMARY", json.dumps(summ))
