"""
Third pre-registered batch (prereg3.json): evaluate gaps built from laya_embed.py outputs.
Usage: python experiments/review/laya_gaps.py [cuda:0]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from heldout import run  # noqa: E402
import common as C, hub_common as H  # noqa: E402
from ladder import hubstats  # noqa: E402

E, L, K = ROOT / "results/emb_cache", ROOT / "results/emb_cache/laya", ROOT / "results/emb_cache/coco4k"


def fl(tag, s):
    p = L / f"flores_{tag}_{s}.pt"
    return torch.load(p) if p.exists() else torch.load(E / f"{tag}__eng_Latn__{s}.pt")


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    I = {"laya-tower": torch.load(L / "coco_laya_tower.pt"), "laya-connector": torch.load(L / "coco_laya_connector.pt"),
         "dinov2": torch.load(K / "facebook_dinov2-base.pt")}
    T = {t: torch.load(K / f"{m}.pt") for t, m in (("bert", "bert-base-uncased"), ("e5", "intfloat_e5-base-v2"),
                                                     ("mpnet", "sentence-transformers_all-mpnet-base-v2"))}
    gaps = [(f"laya/{a}->{b}", "image-text", I[a][:3000], T[b][:3000], I[a][3000:], T[b][3000:])
            for a in ("laya-tower", "laya-connector") for b in T]
    gaps.append(("laya/laya-tower->dinov2", "image-image", I["laya-tower"][:3000], I["dinov2"][:3000], I["laya-tower"][3000:], I["dinov2"][3000:]))
    TX = {"laya-modernbert": "laya-modernbert", "modernbert-base": "modernbert-base-large", "qwen3": "qwen3-0.6b",
          "bert": "bert-base-uncased", "e5": "intfloat_multilingual-e5-base"}
    txt = [("modernbert-base", "laya-modernbert", "fine-tune"), ("laya-modernbert", "bert", "cross-arch"),
           ("laya-modernbert", "e5", "cross-arch"), ("qwen3", "bert", "cross-arch"), ("qwen3", "e5", "cross-arch"),
           ("qwen3", "laya-modernbert", "cross-arch")]
    for a, b, reg in txt:
        gaps.append((f"laya/{a}->{b}", reg, fl(TX[a], "dev"), fl(TX[b], "dev"), fl(TX[a], "devtest"), fl(TX[b], "devtest")))
    out = ROOT / "results/review/laya_gaps.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for name, reg, Xtr, Ytr, Xte, Yte in gaps:
        if name in res:
            continue
        Xtr, Ytr, Xte, Yte = (t.float() for t in (Xtr, Ytr, Xte, Yte))
        r = run(Xtr, Ytr, Xte, Yte, device); r["regime"] = reg
        rot = C.ortho_map(Xtr, Ytr); ccav, _, _ = H.cca_val_select(Xtr, Ytr)
        r["hub"] = {"rotation": hubstats(rot, Xte, Yte)["skew"], "centred": hubstats(H.centred_map(rot, Xtr, Ytr), Xte, Yte)["skew"],
                    "cca": hubstats(ccav, Xte, Yte)["skew"]}
        res[name] = r; out.write_text(json.dumps(res, indent=1))
        print(name, {k: round(v, 3) for k, v in r["p1"].items()}, {k: round(v, 2) for k, v in r["hub"].items()}, "ID", round(r["diag"]["ID_train"], 3), flush=True)


if __name__ == "__main__":
    main()
