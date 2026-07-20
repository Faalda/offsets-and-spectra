"""
Cross-modal image-text alignment with CLIP (ViT-B/32).

We embed MSCOCO test images with the CLIP image encoder and captions with the
CLIP text encoder, then ask whether a linear map (ortho, CCA, linear-ls) can
bridge the gap.  Saves results/crossmodal_clip/summary.json.

CLIP is jointly trained to align image and text spaces, so this tests whether
joint training makes the cross-modal gap linearly bridgeable — in contrast to
independently-trained speech/text encoders where linear geometry fails.
"""
from __future__ import annotations
import sys, json, io
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 1000   # pairs for fitting the map
N_TEST  = 500    # pairs for evaluation
OUT_DIR = ROOT / "results" / "crossmodal_clip"

DATASETS = {
    "mscoco":   ("clip-benchmark/wds_mscoco_captions",  "test"),
    "flickr30k": ("clip-benchmark/wds_flickr30k",        "test"),
    "flickr8k":  ("clip-benchmark/wds_flickr8k",         "test"),
}


# ── alignment helpers ──────────────────────────────────────────────────────────

def p1(M, Y):
    a = F.normalize(M.float(), dim=-1)
    b = F.normalize(Y.float(), dim=-1)
    idx = torch.arange(a.shape[0])
    return (( a @ b.T).argmax(1) == idx).float().mean().item()


def run_ortho(Xtr, Ytr, Xte, Yte):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return p1(Xte @ R, Yte)


def run_cca(Xtr, Ytr, Xte, Yte, lam=0.1):
    n = Xtr.shape[0]; mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]; I = torch.eye(d, device=Xc.device)
    Cxx = Xc.T @ Xc / n + lam * I
    Cyy = Yc.T @ Yc / n + lam * I
    Cxy = Xc.T @ Yc / n
    def isq(C):
        w, V = torch.linalg.eigh(C)
        return V @ torch.diag(w.clamp(min=1e-6) ** -0.5) @ V.T
    Mx, My = isq(Cxx), isq(Cyy)
    U, _, Vt = torch.linalg.svd(Mx @ Cxy @ My)
    return p1((Xte - mx) @ (Mx @ U), (Yte - my) @ (My @ Vt.T))


def run_linear_ls(Xtr, Ytr, Xte, Yte):
    W, _, _, _ = torch.linalg.lstsq(Xtr, Ytr)
    return p1(Xte @ W, Yte)


def run_mlp(Xtr, Ytr, Xte, Yte, h=512, steps=400):
    d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, d))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = (1 - F.cosine_similarity(net(Xtr), Ytr, dim=-1)).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return p1(net(Xte), Yte)


def anisotropy(X, Y):
    def spec(Z):
        Zc = Z - Z.mean(0)
        C = Zc.T @ Zc / Z.shape[0]
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-9)
        ev = ev / ev.sum()
        return ev.sort(descending=True).values
    return (spec(X) - spec(Y)).abs().sum().item()


# ── embedding ──────────────────────────────────────────────────────────────────

def embed_clip(dataset_name: str, hf_name: str, split: str,
               n_train=N_TRAIN, n_test=N_TEST, processor=None, model=None):
    from datasets import load_dataset

    cache_path = ROOT / "results" / "emb_cache" / f"clip_{dataset_name}.pt"
    if cache_path.exists():
        print(f"  Loading cached {dataset_name} embeddings")
        cache = torch.load(cache_path)
        return cache["img"], cache["txt"]

    from PIL import Image
    ds = load_dataset(hf_name, split=split, streaming=True)
    needed = n_train + n_test
    img_embs, txt_embs = [], []
    with torch.no_grad():
        for row in ds:
            if len(img_embs) >= needed:
                break
            try:
                img = row["jpg"].convert("RGB")
                cap = row["txt"].split("\n")[0].strip()
                inputs = processor(text=[cap], images=[img],
                                   return_tensors="pt", padding=True,
                                   truncation=True).to(DEVICE)
                out = model(**inputs)
                img_embs.append(F.normalize(out.image_embeds, dim=-1).cpu())
                txt_embs.append(F.normalize(out.text_embeds,  dim=-1).cpu())
                if len(img_embs) % 200 == 0:
                    print(f"  {len(img_embs)}/{needed}")
            except Exception:
                continue

    X = torch.cat(img_embs)
    Y = torch.cat(txt_embs)
    torch.save({"img": X, "txt": Y}, cache_path)
    return X, Y


# ── main ───────────────────────────────────────────────────────────────────────

def run_dataset(name, hf_name, split, processor, model):
    print(f"\n── {name} ──")
    X, Y = embed_clip(name, hf_name, split, processor=processor, model=model)
    n = X.shape[0]
    n_tr = min(N_TRAIN, int(n * 0.67))
    n_te = n - n_tr
    if n_te < 50:
        print(f"  WARNING: only {n_te} test samples, skipping {name}")
        return None
    print(f"  shape: {tuple(X.shape)}  n_train={n_tr}  n_test={n_te}")
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xte, Yte = X[n_tr:], Y[n_tr:]
    identity = p1(Xte, Yte)
    ortho    = run_ortho(Xtr, Ytr, Xte, Yte)
    cca      = run_cca(Xtr, Ytr, Xte, Yte)
    lin_ls   = run_linear_ls(Xtr, Ytr, Xte, Yte)
    mlp      = run_mlp(Xtr, Ytr, Xte, Yte)
    aniso    = anisotropy(Xtr, Ytr)
    print(f"  aniso={aniso:.3f}  identity={identity:.3f}  "
          f"ortho={ortho:.3f}  cca={cca:.3f}  lin-ls={lin_ls:.3f}  mlp={mlp:.3f}")
    return dict(dataset=name, aniso=aniso, identity=identity,
                ortho=ortho, cca=cca, linear_ls=lin_ls, mlp=mlp,
                n_train=n_tr, n_test=n_te)


def main():
    from transformers import CLIPModel, CLIPProcessor
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP model (openai/clip-vit-base-patch32)…")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()

    results = []
    for name, (hf_name, split) in DATASETS.items():
        r = run_dataset(name, hf_name, split, processor, model)
        if r is not None:
            results.append(r)

    print("\n── Summary ──")
    print(f"{'dataset':<12} {'aniso':>6} {'identity':>9} {'ortho':>7} {'cca':>7} {'lin-ls':>7} {'mlp':>7}")
    for r in results:
        print(f"{r['dataset']:<12} {r['aniso']:>6.3f} {r['identity']:>9.3f} "
              f"{r['ortho']:>7.3f} {r['cca']:>7.3f} {r['linear_ls']:>7.3f} {r['mlp']:>7.3f}")

    json.dump(results, open(OUT_DIR / "summary.json", "w"), indent=2)
    print(f"\nSaved to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
