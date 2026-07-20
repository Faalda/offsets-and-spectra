"""
Stage-4 task: cross-modal speech↔text alignment on SLURP-TN.

Each utterance is a natural (speech, text) pair. A frozen speech encoder
(wav2vec2-large) and a frozen text encoder (XLM-R large) — both 1024-d — produce
two manifolds separated by the *modality gap*. We ask the Stage-2 question one
level up: is the speech↔text gap rigid, and does an orthogonal map close it
better than unconstrained / low-rank (LoRA-style) maps?

This is the method's predicted sweet spot: a genuine cross-modal alignment, not a
single-task fine-tune. Embeddings are cached to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

_CACHE = Path(__file__).resolve().parent.parent / "results" / "emb_cache"
SR = 16000
MAX_SAMPLES = SR * 10          # cap utterances at 10 s


@torch.no_grad()
def _embed_audio(arrays, model_id, device) -> torch.Tensor:
    """Mean-pooled, L2-normalised utterance embeddings from a frozen speech encoder.

    Whisper (a semantic ASR encoder) and wav2vec2 (an acoustic encoder) take
    different feature front-ends; we branch on the model id."""
    is_whisper = "whisper" in model_id.lower()
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    encoder = model.encoder if is_whisper else model
    out = []
    for arr in arrays:
        a = torch.as_tensor(arr, dtype=torch.float32)[:MAX_SAMPLES]
        feats = fe(a.numpy(), sampling_rate=SR, return_tensors="pt")
        x = (feats.input_features if is_whisper else feats.input_values).to(device)
        h = encoder(x).last_hidden_state            # (1, T, D)
        out.append(torch.nn.functional.normalize(h.mean(1), dim=-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def _pca_project(train, others, dim):
    """Fit PCA on `train` (rows = samples), project train+others to `dim`, renormalise."""
    mu = train.mean(0)
    _, _, Vt = torch.linalg.svd(train - mu, full_matrices=False)
    comp = Vt[:dim]                                  # (dim, D)
    proj = lambda Z: torch.nn.functional.normalize((Z - mu) @ comp.T, dim=-1)
    return proj(train), [proj(o) for o in others]


@torch.no_grad()
def _embed_text(texts, model_id, device) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def _cached(kind, split, model_id, device, loader):
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"slurptn__{kind}__{model_id.replace('/', '_')}__{split}.pt"
    if path.exists():
        return torch.load(path)
    emb = loader()
    torch.save(emb, path)
    return emb


@dataclass
class CrossModalData:
    X_train: torch.Tensor; Y_train: torch.Tensor       # speech, text (train)
    X_val: torch.Tensor;   Y_val: torch.Tensor
    X_test: torch.Tensor;  Y_test: torch.Tensor
    dim: int


def load_crossmodal(speech_model="jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
                    text_model="xlm-roberta-large",
                    text_col="tun_transcription", device="cpu") -> CrossModalData:
    from datasets import load_dataset
    splits = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset("Elyadata/SLURP-TN", split=split)
        X = _cached("speech", split, speech_model, device,
                    lambda ds=ds: _embed_audio([a["array"] for a in ds["audio"]], speech_model, device))
        Y = _cached("text", split, text_model, device,
                    lambda ds=ds: _embed_text([str(t) for t in ds[text_col]], text_model, device))
        splits[split] = (X, Y)
    (Xtr, Ytr), (Xv, Yv), (Xte, Yte) = splits["train"], splits["validation"], splits["test"]
    # Orthogonal/linear alignment needs matching dims; PCA-reduce the larger side
    # (fit on train only) down to the smaller. A fixed linear preprocessing step.
    if Xtr.shape[1] != Ytr.shape[1]:
        d = min(Xtr.shape[1], Ytr.shape[1])
        if Xtr.shape[1] > d:
            Xtr, (Xv, Xte) = _pca_project(Xtr, [Xv, Xte], d)
        else:
            Ytr, (Yv, Yte) = _pca_project(Ytr, [Yv, Yte], d)
    return CrossModalData(Xtr, Ytr, Xv, Yv, Xte, Yte, Xtr.shape[1])
