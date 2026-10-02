"""
Direct row-order verification of the LibriSpeech caches (added 2026-10-02).

For a spread of dataset indices, re-extract the audio and the transcript from openslr/librispeech_asr
(clean, test) with the SAME functions that built the caches, and compare each fresh embedding with the cached
row at that index.  If a cache is in dataset order, the fresh embedding of item i has cosine ~1 with cached row i
and clearly lower cosine with cached rows j != i (control reported).

Usage: python experiments/review/librispeech_row_check.py [cuda:1]
Writes: results/review/librispeech_row_check.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.crossmodal import _audio_arrays, _embed_audio, _embed_text  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
E = ROOT / "results" / "emb_cache"
cache_audio = torch.load(E / "librispeech__speech-validframes-fp32__openai_whisper-large-v3__test.pt").float()
cache_text = torch.load(E / "librispeech__text__xlm-roberta-large__test.pt").float()
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
n = len(ds)
idx = sorted(set(np.linspace(0, n - 1, 40).astype(int).tolist()))
sub = ds.select(idx)
fresh_audio = _embed_audio(_audio_arrays(sub), "openai/whisper-large-v3", device).float().cpu()
fresh_text = _embed_text(list(sub["text"]), "xlm-roberta-large", device).float().cpu()
# Raw cosines of mean-pooled XLM-R embeddings are ~0.99 between ANY two sentences (strong anisotropy), so a raw
# "own-row" cosine cannot show correct ordering.  Both sides are therefore centred with the cached mean first.
def cos(A, B, mu):
    A, B = torch.nn.functional.normalize(A - mu, dim=1), torch.nn.functional.normalize(B - mu, dim=1)
    return A @ B.T
out = {"n_dataset": n, "n_cache_audio": len(cache_audio), "n_cache_text": len(cache_text), "indices": idx}
for name, fresh, cache in (("audio", fresh_audio, cache_audio), ("text", fresh_text, cache_text)):
    S = cos(fresh, cache[idx], cache.mean(0))        # fresh item i vs cached rows at the sampled indices (centred)
    own = S.diag()
    off = S[~torch.eye(len(idx), dtype=bool)]
    out[name] = {"own_row_cosine_min": float(own.min()), "own_row_cosine_mean": float(own.mean()),
                 "other_rows_cosine_max": float(off.max()), "other_rows_cosine_mean": float(off.mean()),
                 "argmax_matches_own_row": int((S.argmax(1) == torch.arange(len(idx))).sum()), "n_checked": len(idx)}
(ROOT / "results/review/librispeech_row_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
