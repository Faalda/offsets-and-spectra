"""
Stage-2 task: cross-lingual sentence-embedding alignment on FLORES-200.

FLORES-200 is fully index-aligned across languages, so loading two language
configs and pairing by row gives parallel sentence pairs. We embed each language
with a *frozen* multilingual encoder (multilingual-e5-base) and treat the two
resulting point clouds as two manifolds to align.

The historical result this generalises: the optimal map between independently
learned embedding spaces is orthogonal (Procrustes / MUSE). Here we ask whether
the residual gap between two languages *inside a shared encoder* is also rigid.

Embeddings are cached to disk, so only the first run pays the encoding cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

_CACHE = Path(__file__).resolve().parent.parent / "results" / "emb_cache"
_DEFAULT_ENCODER = "intfloat/multilingual-e5-base"


def _default_prefix(encoder: str) -> str:
    """e5 models require a 'query: ' / 'passage: ' prefix; plain BERTs don't."""
    return "query: " if "e5" in encoder.lower() else ""


@torch.no_grad()
def embed_sentences(sentences, encoder: str, device, batch_size: int = 64,
                    prefix: str | None = None) -> torch.Tensor:
    """Mean-pooled, L2-normalised sentence embeddings from any HF encoder."""
    from transformers import AutoModel, AutoTokenizer

    if prefix is None:
        prefix = _default_prefix(encoder)
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = [f"{prefix}{s}" for s in sentences[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state              # (B, T, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def _cached_embed(lang: str, split: str, encoder: str, device) -> torch.Tensor:
    _CACHE.mkdir(parents=True, exist_ok=True)
    tag = encoder.replace("/", "_")
    path = _CACHE / f"{tag}__{lang}__{split}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    ds = load_dataset("Muennighoff/flores200", lang, split=split, trust_remote_code=True)
    emb = embed_sentences(list(ds["sentence"]), encoder, device)
    torch.save(emb, path)
    return emb


@dataclass
class CrossLingualData:
    X_train: torch.Tensor   # source embeddings (train pairs)
    Y_train: torch.Tensor   # target embeddings (train pairs)
    X_test: torch.Tensor
    Y_test: torch.Tensor
    src: str
    tgt: str
    dim: int


def load_crosslingual(src: str = "eng_Latn", tgt: str = "arb_Arab",
                      encoder: str = _DEFAULT_ENCODER, device="cpu",
                      encoder_tgt: str | None = None) -> CrossLingualData:
    """FLORES dev split as train pairs, devtest as held-out test pairs.

    encoder_tgt: optional second encoder for the target side. When it differs
    from `encoder` the two manifolds come from *independent* models — the natural
    cross-encoder gap (Stage 2b), where it is an open question whether the optimal
    map is rigid. Requires both encoders to share the same hidden dimension."""
    enc_t = encoder_tgt or encoder
    Xtr = _cached_embed(src, "dev", encoder, device)
    Ytr = _cached_embed(tgt, "dev", enc_t, device)
    Xte = _cached_embed(src, "devtest", encoder, device)
    Yte = _cached_embed(tgt, "devtest", enc_t, device)
    if Xtr.shape[1] != Ytr.shape[1]:
        raise ValueError(
            f"Encoder dims differ ({Xtr.shape[1]} vs {Ytr.shape[1]}). "
            "Orthogonal alignment needs equal dims; pick encoders with the same hidden size.")
    return CrossLingualData(Xtr, Ytr, Xte, Yte, src, tgt, Xtr.shape[1])
