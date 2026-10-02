"""
Embed FLORES English (dev, devtest) with the frozen list of new source encoders (scale_config.py).
Same routine and cache naming as tasks/crosslingual.py (mean-pooled, L2-normalised, max_length 128);
adds a pad-token fix for decoder models.  Skips anything already cached.

Usage: python experiments/review/scale_embed.py [cuda:1]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from scale_config import NEW_SOURCES  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"


@torch.no_grad()
def embed(sentences, encoder, device, batch_size=64):
    from transformers import AutoModel, AutoTokenizer
    prefix = "query: " if "e5" in encoder.lower() else ""
    tok = AutoTokenizer.from_pretrained(encoder)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = [f"{prefix}{s}" for s in sentences[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    from datasets import load_dataset
    sents = {sp: list(load_dataset("Muennighoff/flores200", "eng_Latn", split=sp)["sentence"])
             for sp in ("dev", "devtest")}
    print({k: len(v) for k, v in sents.items()}, flush=True)
    for enc, note in NEW_SOURCES:
        tag = enc.replace("/", "_")
        paths = {sp: CACHE / f"{tag}__eng_Latn__{sp}.pt" for sp in ("dev", "devtest")}
        if all(p.exists() for p in paths.values()):
            print("cached ", enc, flush=True); continue
        t0 = time.time()
        try:
            for sp, p in paths.items():
                if not p.exists():
                    torch.save(embed(sents[sp], enc, device), p)
            d = torch.load(paths["dev"]).shape[1]
            print(f"done   {enc:<45} d={d:<5} {time.time()-t0:5.0f}s", flush=True)
        except Exception as e:  # record and continue; failures are reported, not hidden
            print(f"FAILED {enc}: {type(e).__name__}: {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
