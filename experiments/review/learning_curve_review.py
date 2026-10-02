"""Reviewer-requested sample-size sweeps with validation-tuned CCA.

Uses cached e5->German-BERT and DINOv2->BERT embeddings. Each point averages
three fixed subsamples; test sets remain fixed. Writes
results/review/learning_curve_review.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "review"))

from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402
import hub_common as H  # noqa: E402

SEEDS = (0, 1, 2)


def summarize(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values)), "values": values}


def curve(Xtr, Ytr, Xte, Yte, sizes, device):
    Xtr, Ytr, Xte, Yte = (tensor.float().to(device) for tensor in (Xtr, Ytr, Xte, Yte))
    output = {}
    for size in sizes:
        rows = []
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            index = torch.randperm(len(Xtr), generator=generator)[:size].to(device)
            X, Y = Xtr[index], Ytr[index]
            centred = H.centred_map(C.ortho_map(X, Y), X, Y)
            cca, selected_lambda, _ = H.cca_val_select(X, Y)
            rows.append({
                "centred": C.evalmap(centred, Xte, Yte),
                "cca": C.evalmap(cca, Xte, Yte),
                "selected_lambda": selected_lambda,
            })
        output[str(size)] = {
            "centred": summarize([row["centred"] for row in rows]),
            "cca": summarize([row["cca"] for row in rows]),
            "gain": summarize([row["cca"] - row["centred"] for row in rows]),
            "selected_lambda": [row["selected_lambda"] for row in rows],
        }
        print(size, output[str(size)], flush=True)
    return output


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    text = load_crosslingual(
        "eng_Latn", "deu_Latn", "intfloat/multilingual-e5-base", "cpu",
        encoder_tgt="bert-base-german-cased",
    )
    cache = ROOT / "results" / "emb_cache" / "coco4k"
    image = torch.load(cache / "facebook_dinov2-base.pt")
    caption = torch.load(cache / "bert-base-uncased.pt")
    output = {
        "text/e5->de-bert": curve(
            text.X_train, text.Y_train, text.X_test, text.Y_test,
            (100, 200, 500, 997), device,
        ),
        "image-text/dinov2->bert": curve(
            image[:3000], caption[:3000], image[3000:], caption[3000:],
            (250, 500, 1000, 2000, 3000), device,
        ),
        "seeds": list(SEEDS),
    }
    path = ROOT / "results" / "review" / "learning_curve_review.json"
    path.write_text(json.dumps(output, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()