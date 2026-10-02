"""
Freeze the pre-registered hypotheses and prediction models BEFORE any held-out gap is
embedded.  Writes results/review/prereg.json with sha256 of the dev result files.

H1  centring gain  ~ mean dominance of the source (mean_dom_x): Spearman > 0 (one-sided p<.05)
H2  whitening gain beyond centring ~ spectral mismatch A: Spearman > 0 (one-sided p<.05)
H3  rule "identity if ID>=0.5 else CCA(val lambda)" has mean regret <= 0.01 on held-out gaps,
    candidates {identity, rotation, centred rotation, CCA(val), MLP-InfoNCE}
H4  1-D linear fits frozen here (dev data) predict held-out g_centre and g_whiten with lower
    MAE than the frozen dev mean
"""
import hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results" / "review"
D = json.loads((R / "decomposition.json").read_text())["rows"]
within = [r for r in D if r["regime"] in ("cross-lingual", "cross-arch")]
fit = {}
for g, x in (("g_centre", "mean_dom_x"), ("g_whiten", "A")):
    X = np.array([r[x] for r in within]); Y = np.array([r[g] for r in within])
    c = np.polyfit(X, Y, 1)
    fit[g] = {"predictor": x, "slope": float(c[0]), "intercept": float(c[1]), "dev_mean": float(Y.mean()), "n_dev": len(within)}
pre = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
       "hypotheses": __doc__,
       "fits": fit, "thresholds": {"ID": 0.5, "tie": 0.005, "H3_max_mean_regret": 0.01},
       "dev_file_sha256": {f: hashlib.sha256((R / f).read_bytes()).hexdigest()
                           for f in ("decomposition.json", "hub_dev.json", "selection.json")},
       "heldout_gaps_planned": ["mpnet-multilingual -> 17 monolingual BERTs (new source family)",
                                "e5 -> new languages: el, he, ro, hu, da, cs, uk, hi (where a 768-d monolingual BERT loads)",
                                "SLURP speech: whisper->wav2vec2, whisper->XLM-R, wav2vec2->XLM-R (new domain)"]}
(R / "prereg.json").write_text(json.dumps(pre, indent=1)); print(json.dumps(pre["fits"], indent=1)); print(pre["frozen_at"])
