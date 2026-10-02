"""
R2-W2 / rebuttal Q2: larger source encoders at NATIVE dimension, no PCA.

Source: gte-Qwen2-1.5B-instruct (d=1536) and multilingual-e5-large (d=1024), English
FLORES.  Target: the 12 monolingual 768-d BERTs.  Unequal dimensions are handled
natively: Procrustes gives a semi-orthogonal d_x x d_y map (U V^T from the thin SVD
of X^T Y), CCA works on any pair of dimensions, ridge is d_x x d_y.
Also re-runs the PCA-768 variant (PCA fitted on TRAIN source only) for comparison,
and reports wall-clock of every fit on CPU and GPU, plus the diagnostic.

Usage: python experiments/review/native_dim.py [cuda:0]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
from benchmark import XL  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
SOURCES = {"gte-Qwen2-1.5B": "Alibaba-NLP_gte-Qwen2-1.5B-instruct", "e5-large": "intfloat_multilingual-e5-large"}


def ld(tag, lang, split):
    return torch.load(CACHE / f"{tag}__{lang}__{split}.pt").float()


def timed(fn, *a, device="cpu"):
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); m = fn(*a)
    if device != "cpu":
        torch.cuda.synchronize()
    return m, time.perf_counter() - t0


def pca_fit(X, k):
    mu = X.mean(0); _, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T
    return lambda Z: (Z - mu) @ P


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for sname, stag in SOURCES.items():
        Xtr_all, Xte_all = ld(stag, "eng_Latn", "dev"), ld(stag, "eng_Latn", "devtest")
        res[sname] = {}
        for tgt, enc, lab in XL[:12]:
            Ytr, Yte = ld(enc.replace("/", "_"), tgt, "dev"), ld(enc.replace("/", "_"), tgt, "devtest")
            r = {"dx": Xtr_all.shape[1], "dy": Ytr.shape[1], "native": {}, "pca768": {}, "time_s": {}}
            for dev in ("cpu", device):
                Xtr, Ytr_d, Xte, Yte_d = (t.to(dev) for t in (Xtr_all, Ytr, Xte_all, Yte))
                for k, f in {"ortho": C.ortho_map, "cca": C.cca_map,
                             "ridge(lam=1e-2)": lambda a, b: C.ridge_map(a, b, 1e-2)}.items():
                    m, t = timed(f, Xtr, Ytr_d, device=dev)
                    r["time_s"][f"{k}@{dev}"] = t
                    if dev == "cpu":
                        r["native"][k] = C.evalmap(m, Xte, Yte_d)
                _, t = timed(C.anisotropy, Xtr, Ytr_d, device=dev); r["time_s"][f"diagnostic@{dev}"] = t
            r["native"]["ridge-tuned"] = C.evalmap(C.ridge_tuned(Xtr_all, Ytr)[0], Xte_all, Yte)
            r["native"]["mlp-nce"] = C.evalmap(C.mlp_tuned(Xtr_all, Ytr, device=device)[0], Xte_all, Yte)
            r["aniso"] = C.anisotropy(Xtr_all, Ytr)
            P = pca_fit(Xtr_all, 768)
            for k, f in {"ortho": C.ortho_map, "cca": C.cca_map}.items():
                r["pca768"][k] = C.evalmap(f(P(Xtr_all), Ytr), P(Xte_all), Yte)
            res[sname][lab] = r
            print(sname, lab, {k: round(v, 3) for k, v in r["native"].items()},
                  "pca", {k: round(v, 3) for k, v in r["pca768"].items()},
                  {k: round(v, 3) for k, v in r["time_s"].items()}, flush=True)
    out = ROOT / "results" / "review" / "native_dim.json"
    out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
