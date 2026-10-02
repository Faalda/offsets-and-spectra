"""Reproduce final manuscript sensitivity statistics from saved result records.

This script performs no new model fitting. It distinguishes preregistered tests
from post hoc sensitivity checks and writes results/review/final_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "review"


def rho(x, y):
    r, p = spearmanr(x, y)
    return {"rho": float(r), "p_two_sided": float(p), "n": len(x)}


def main():
    intent = json.loads((R / "transfer_v2.json").read_text())["_avg"]
    sentiment = json.loads((R / "sentiment.json").read_text())["_avg"]
    methods = ["ortho", "ortho-centred", "cca", "cca-k", "ridge", "mlp-nce"]
    held = json.loads((R / "heldout_eval.json").read_text())
    rows = held["rows"]
    h1 = held["results"]["H1"]
    h2 = held["results"]["H2"]
    families = {}
    for row in rows:
        families.setdefault(row["mean_dom_x"], []).append(row["g_centre"])
    family_x = list(families)
    family_y = [float(np.mean(v)) for v in families.values()]
    no_slurp = [r for r in rows if not r["gap"].startswith("slurp/")]
    gains = np.array([r["g_whiten"] for r in json.loads((R / "decomposition.json").read_text())["rows"]
                      if r["regime"] in ("cross-lingual", "cross-arch")])
    out = {
        "transfer_rank_agreement": {
            "methods": methods,
            "massive": rho([intent[f"flores_p1_{m}"] for m in methods], [intent[m] for m in methods]),
            "sentiment": rho([sentiment[f"p1_{m}"] for m in methods], [sentiment[m] for m in methods]),
        },
        "H1_preregistered_gap_level": h1,
        "H1_posthoc_source_family_means": {
            "distinct_source_values": len(family_x),
            "values": [{"mean_dominance": float(x), "mean_centring_gain": float(y), "gaps": len(families[x])}
                       for x, y in zip(family_x, family_y)],
            "spearman": rho(family_x, family_y),
        },
        "H1_posthoc_without_slurp": rho([r["mean_dom_x"] for r in no_slurp], [r["g_centre"] for r in no_slurp]),
        "H2_preregistered": h2,
        "within_text_whitening_gain": {
            "n": int(len(gains)),
            "median": float(np.median(gains)),
            "q1": float(np.percentile(gains, 25)),
            "q3": float(np.percentile(gains, 75)),
            "minimum": float(gains.min()),
            "maximum": float(gains.max()),
            "within_abs_0_01": int(np.sum(np.abs(gains) < 0.01)),
        },
    }
    (R / "final_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
