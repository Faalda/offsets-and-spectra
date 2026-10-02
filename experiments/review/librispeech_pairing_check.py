"""
Pairing check for the speech caches (added 2026-10-02).

Question: are the rows of an audio cache aligned with the rows of its transcript cache?  Retrieval near chance is
what a row-order mismatch would produce, so it cannot by itself be read as "no alignment signal".

Test: on the first N paired items, centre and L2-normalise each side, compute all pairwise cosine similarities,
and take the Spearman correlation between the audio and text similarity structures over the upper triangle.
If rows are paired, utterances with similar content/length should be similar on both sides (positive);
if the rows are misaligned the value is ~0.  Control: the same statistic after randomly permuting the text rows.
Also reported: the correlation between two audio caches of the same utterances (same row order => clearly > 0).

Usage: python experiments/review/librispeech_pairing_check.py     Writes results/review/librispeech_pairing_check.json
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results" / "emb_cache"
N = 800
load = lambda n: torch.load(E / n, map_location="cpu").float()
sim = lambda X: (torch.nn.functional.normalize(X - X.mean(0), dim=1) @ torch.nn.functional.normalize(X - X.mean(0), dim=1).T).numpy()
CASES = {
    "librispeech_padded_pooling_old": ("librispeech__speech__openai_whisper-large-v3__test.pt", "librispeech__text__xlm-roberta-large__test.pt"),
    "librispeech_valid_frames_new": ("librispeech__speech-validframes-fp32__openai_whisper-large-v3__test.pt", "librispeech__text__xlm-roberta-large__test.pt"),
    "slurp_valid_frames": ("slurptn__speech-validframes-fp32__openai_whisper-large-v3__test.pt", "slurptn__text__xlm-roberta-large__test.pt"),
}
rng = np.random.default_rng(0)
out = {"n_items": N}
iu = np.triu_indices(N, 1)
sims = {}
for name, (a, t) in CASES.items():
    A, T = load(a)[:N], load(t)[:N]
    Sa, St = sim(A), sim(T); sims[name] = Sa
    perm = rng.permutation(N)
    out[name] = {"audio_vs_text_spearman": float(spearmanr(Sa[iu], St[iu])[0]),
                 "shuffled_text_control": float(spearmanr(Sa[iu], St[np.ix_(perm, perm)][iu])[0])}
out["old_audio_vs_new_audio_spearman"] = float(spearmanr(sims["librispeech_padded_pooling_old"][iu], sims["librispeech_valid_frames_new"][iu])[0])
(ROOT / "results/review/librispeech_pairing_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
