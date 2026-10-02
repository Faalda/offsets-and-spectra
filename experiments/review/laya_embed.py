"""
Embed for the third pre-registered batch (prereg3.json). Run with the project venv, where the
`laya` package loads:  python experiments/review/laya_embed.py
Outputs results/emb_cache/laya/*.pt
  COCO 4k images: laya-vision SigLIP tower (mean over patch tokens) and connector output (mean over
                  the 64 image tokens, i.e. the language model's input space)
  FLORES eng dev/devtest: Laya fine-tuned ModernBERT-large, base ModernBERT-large, Qwen3-0.6B
                  (all mean-pooled last hidden state)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "emb_cache" / "laya"; OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda:0"


@torch.no_grad()
def images():
    import laya
    if (OUT / "coco_laya_connector.pt").exists():
        return
    imgs, _ = torch.load(ROOT / "results/emb_cache/coco4k/pairs.pt", weights_only=False)
    m = laya.load_vlm("thaitea/laya-vision", device=DEV).model.eval()
    tower, conn = [], []
    for i in range(len(imgs)):   # the Laya preprocessor stacks only equal-size images
        batch = [torch.from_numpy(np.array(imgs[i]))]
        pv, pam = m.prep.pixel_values(batch, device=m.encoder.device, dtype=m.encoder.dtype)
        f = m.encode_images(pv, pam)                       # [n_img, 64, 576]
        conn.append(f.float().mean(1).cpu())
        o = m.encoder.vision_model(pixel_values=pv.reshape(-1, *pv.shape[-3:]), **m._vision_kw)
        tower.append(o.last_hidden_state.float().mean(1).cpu())
    torch.save(torch.cat(tower), OUT / "coco_laya_tower.pt"); torch.save(torch.cat(conn), OUT / "coco_laya_connector.pt")
    print("images", torch.cat(tower).shape, torch.cat(conn).shape, flush=True)
    del m; torch.cuda.empty_cache()


@torch.no_grad()
def pool(model, tok, sents, bs=32, max_len=128):
    out = []
    for i in range(0, len(sents), bs):
        e = tok(sents[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        h = model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state.float()
        mk = e["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).cpu())
    return torch.cat(out)


def texts():
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    import laya
    fl = {s: list(load_dataset("Muennighoff/flores200", "eng_Latn", split=s, trust_remote_code=True)["sentence"]) for s in ("dev", "devtest")}
    encs = {}
    ag = laya.load("convaiinnovations/laya-typed-decisions", device=DEV)
    encs["laya-modernbert"] = (ag.model.encoder.eval(), ag.tok)
    for name, mid in (("modernbert-base-large", "answerdotai/ModernBERT-large"), ("qwen3-0.6b", "Qwen/Qwen3-0.6B")):
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        encs[name] = (AutoModel.from_pretrained(mid, torch_dtype=torch.float32).to(DEV).eval(), tok)
    for name, (m, tok) in encs.items():
        for s, sents in fl.items():
            p = OUT / f"flores_{name}_{s}.pt"
            if not p.exists():
                torch.save(pool(m, tok, sents), p)
        print(name, "done", flush=True)


if __name__ == "__main__":
    images(); texts()
