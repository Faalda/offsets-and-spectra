"""Paired language-bootstrap intervals requested for transfer comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "review"


def interval(values, seed=0, samples=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_languages": len(values),
    }


def main():
    massive = json.loads((RESULTS / "transfer_v2.json").read_text())
    sentiment = json.loads((RESULTS / "sentiment.json").read_text())
    massive_languages = [key for key, value in massive.items() if not key.startswith("_") and isinstance(value, dict)]
    sentiment_languages = [key for key, value in sentiment.items() if not key.startswith("_") and isinstance(value, dict)]
    output = {
        "massive_cca_k_minus_centred": interval([
            massive[language]["cca-k"] - massive[language]["ortho-centred"]
            for language in massive_languages
        ]),
        "sentiment_ridge_minus_mlp": interval([
            sentiment[language]["ridge"] - sentiment[language]["mlp-nce"]
            for language in sentiment_languages
        ]),
    }
    path = RESULTS / "transfer_uncertainty.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()