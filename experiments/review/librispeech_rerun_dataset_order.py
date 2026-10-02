"""
LibriSpeech rerun with a transcript cache in DATASET ORDER (added 2026-10-02).

Why: librispeech_row_check.py / the similarity-structure checks showed that the corrected (valid-frame) AUDIO cache is
in dataset order (fresh audio embeddings match their own cached rows, 40/40) but the cached TEXT file
librispeech__text__xlm-roberta-large__test.pt is NOT (its similarity structure matches a shuffled control and its rows
have norm ~31, so it comes from an older pipeline).  The earlier corrected rerun therefore paired mismatched rows.

This script re-embeds the transcripts of openslr/librispeech_asr (clean, test) in dataset order with the same function
used for the other caches (tasks.crossmodal._embed_text), saves them to a NEW cache file, verifies the pairing, and
repeats the five random 80/20 splits of reviewer_checks.corrected_librispeech with the same evaluation code.

Usage: python experiments/review/librispeech_rerun_dataset_order.py [cuda:1]
Writes: results/review/librispeech_dataset_order_rerun.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks.crossmodal import _embed_text  # noqa: E402
import reviewer_checks as RC  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
E = ROOT / "results" / "emb_cache"
new_path = E / "librispeech__text-dataset-order__xlm-roberta-large__test.pt"
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
if new_path.exists():
    text = torch.load(new_path)
else:
    text = _embed_text(list(ds["text"]), "xlm-roberta-large", device).float().cpu()
    torch.save(text, new_path)
speech = torch.load(E / "librispeech__speech-validframes-fp32__openai_whisper-large-v3__test.pt").float()
assert len(speech) == len(text) == len(ds)

# pairing verification: similarity-structure correlation of the paired caches vs a shuffled control
sim = lambda X: (torch.nn.functional.normalize(X - X.mean(0), dim=1) @ torch.nn.functional.normalize(X - X.mean(0), dim=1).T).numpy()
n = 800; iu = np.triu_indices(n, 1); Sa, St = sim(speech[:n]), sim(text[:n])
perm = np.random.default_rng(0).permutation(n)
out = {"pairing_check": {"audio_vs_text_structure_spearman": float(spearmanr(Sa[iu], St[iu])[0]),
                         "shuffled_control": float(spearmanr(Sa[iu], St[np.ix_(perm, perm)][iu])[0])}}
print(out["pairing_check"], flush=True)

res = {}
for seed in range(5):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(speech), generator=g); k = int(0.8 * len(speech)); tr, te = p[:k], p[k:]
    res[f"librispeech/whisper-validframes->xlmr(dataset-order text)/s{seed}"] = RC.evaluate(speech[tr], text[tr], speech[te], text[te])
    pp = res[f"librispeech/whisper-validframes->xlmr(dataset-order text)/s{seed}"]["p1"]
    print(seed, {a: round(b, 3) for a, b in pp.items()}, flush=True)
keys = list(next(iter(res.values()))["p1"].keys())
out["mean_p1_over_5_splits"] = {k: float(np.mean([r["p1"][k] for r in res.values()])) for k in keys}
out["splits"] = res
(ROOT / "results/review/librispeech_dataset_order_rerun.json").write_text(json.dumps(out, indent=1))
print("MEAN over 5 splits:", {a: round(b, 3) for a, b in out["mean_p1_over_5_splits"].items()})
