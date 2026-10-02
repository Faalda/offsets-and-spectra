"""Regenerate LibriSpeech clean/test Whisper embeddings without padding frames."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tasks.crossmodal import _audio_arrays, _embed_audio  # noqa: E402


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="test")
    embeddings = _embed_audio(_audio_arrays(dataset), "openai/whisper-large-v3", device)
    path = ROOT / "results" / "emb_cache" / "librispeech__speech-validframes-fp32__openai_whisper-large-v3__test.pt"
    torch.save(embeddings, path)
    print({"path": str(path), "shape": tuple(embeddings.shape), "dtype": str(embeddings.dtype)})


if __name__ == "__main__":
    main()