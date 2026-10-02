"""
Second pre-registered held-out batch (results/review/prereg2.json), part A.
4,000 COCO train image/caption pairs (first caption) streamed from clip-benchmark/wds_mscoco_captions.
Image encoders (independently trained, no text): DINOv2-base, ViT-B/16 (IN-21k), ConvNeXt-B (IN-22k).
Text encoders (no images): BERT-base, e5-base-v2, all-mpnet-base-v2.
Gaps: 9 image->text + 3 image->image; 3,000 train / 1,000 test.  Evaluation = heldout.run().
Usage: python experiments/review/coco_gaps.py [cuda:0]
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks.crosslingual import embed_sentences  # noqa: E402
from heldout import run  # noqa: E402

CACHE = ROOT / "results" / "emb_cache" / "coco4k"
N = 4000
IMG = {"dinov2": "facebook/dinov2-base", "vit": "google/vit-base-patch16-224-in21k", "convnext": "facebook/convnext-base-224-22k"}
TXT = {"bert": "bert-base-uncased", "e5": "intfloat/e5-base-v2", "mpnet": "sentence-transformers/all-mpnet-base-v2"}


def load_pairs():
    p = CACHE / "pairs.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train", streaming=True)
    imgs, caps = [], []
    for r in itertools.islice(ds, N):
        imgs.append(r["jpg"].convert("RGB")); caps.append(r["txt"].split("\n")[0].strip())
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save((imgs, caps), p)
    return imgs, caps


@torch.no_grad()
def embed_images(imgs, model_id, device, bs=64):
    p = CACHE / f"{model_id.replace('/', '_')}.pt"
    if p.exists():
        return torch.load(p)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id); m = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    for i in range(0, len(imgs), bs):
        o = m(**proc(images=imgs[i:i + bs], return_tensors="pt").to(device))
        z = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
        out.append(z.reshape(z.shape[0], -1).float().cpu())
    del m; torch.cuda.empty_cache()
    E = torch.cat(out); torch.save(E, p)
    return E


def embed_text(caps, model_id, device):
    p = CACHE / f"{model_id.replace('/', '_')}.pt"
    if p.exists():
        return torch.load(p)
    E = embed_sentences(caps, model_id, device); torch.save(E, p)
    return E


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    imgs, caps = load_pairs()
    I = {k: embed_images(imgs, v, device) for k, v in IMG.items()}
    T = {k: embed_text(caps, v, device) for k, v in TXT.items()}
    print({k: tuple(v.shape) for k, v in {**I, **T}.items()}, flush=True)
    out = ROOT / "results" / "review" / "coco_gaps.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    gaps = [(f"coco/{a}->{b}", "image-text", I[a], T[b]) for a in I for b in T] + \
           [(f"coco/{a}->{b}", "image-image", I[a], I[b]) for a, b in itertools.combinations(I, 2)]
    for name, regime, X, Y in gaps:
        if name in res:
            continue
        r = run(X[:3000], Y[:3000], X[3000:], Y[3000:], device); r["regime"] = regime
        # hub skew along the ladder for P2/P3
        import hub_common as H, common as C
        from ladder import hubstats
        rot = C.ortho_map(X[:3000], Y[:3000]); ccav, _, _ = H.cca_val_select(X[:3000], Y[:3000])
        r["hub"] = {"rotation": hubstats(rot, X[3000:], Y[3000:])["skew"],
                    "centred": hubstats(H.centred_map(rot, X[:3000], Y[:3000]), X[3000:], Y[3000:])["skew"],
                    "cca": hubstats(ccav, X[3000:], Y[3000:])["skew"]}
        res[name] = r; out.write_text(json.dumps(res, indent=1))
        print(name, {k: round(v, 3) for k, v in r["p1"].items()}, {k: round(v, 2) for k, v in r["hub"].items()}, flush=True)


if __name__ == "__main__":
    main()
