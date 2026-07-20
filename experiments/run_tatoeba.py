"""
Phase-C second benchmark: cross-encoder alignment on Tatoeba (mteb bitext mining),
to show the orthogonal-beats-linear result is not specific to FLORES.

Per language: English (e5 source) vs monolingual target BERT, 1000 Tatoeba pairs
split 500/500 (fit R / test). Saves results/tatoeba/summary.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dsv.baselines import linear_ls            # noqa: E402
from tasks.crosslingual import embed_sentences  # noqa: E402

SRC_ENC = "intfloat/multilingual-e5-base"
DEVICE = "cuda:0"
N_BOOT = 2000
PAIRS = [
    ("deu-eng", "bert-base-german-cased", "en-de"),
    ("spa-eng", "dccuchile/bert-base-spanish-wwm-cased", "en-es"),
    ("fra-eng", "camembert-base", "en-fr"),
    ("ita-eng", "dbmdz/bert-base-italian-cased", "en-it"),
    ("por-eng", "neuralmind/bert-base-portuguese-cased", "en-pt"),
    ("nld-eng", "GroNLP/bert-base-dutch-cased", "en-nl"),
    ("tur-eng", "dbmdz/bert-base-turkish-cased", "en-tr"),
    ("fin-eng", "TurkuNLP/bert-base-finnish-cased-v1", "en-fi"),
    ("rus-eng", "DeepPavlov/rubert-base-cased", "en-ru"),
    ("ara-eng", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
    ("cmn-eng", "bert-base-chinese", "en-zh"),
    ("pes-eng", "HooshvareLab/bert-base-parsbert-uncased", "en-fa"),
]


def p1_correct(mapped, tgt):
    a, b = F.normalize(mapped, dim=-1), F.normalize(tgt, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).cpu().numpy().astype(np.float64)


def ci(c, rng):
    n = len(c)
    m = np.array([c[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(c.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    from datasets import load_dataset
    rng = np.random.default_rng(0)
    rows = {}
    for cfg, enc_t, label in PAIRS:
        try:
            ds = load_dataset("mteb/tatoeba-bitext-mining", cfg, split="test",
                              trust_remote_code=True)
            eng = list(ds["sentence2"]); tgt = list(ds["sentence1"])
            n = len(eng); h = n // 2
            X = embed_sentences(eng, SRC_ENC, DEVICE)
            Y = embed_sentences(tgt, enc_t, DEVICE)
            Xtr, Ytr, Xte, Yte = X[:h], Y[:h], X[h:], Y[h:]
            Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
            U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
            R = U @ Vt
            maps = {"identity": lambda x: x, "ortho": lambda x: x @ R,
                    "linear-ls": linear_ls(Xtr, Ytr)}
            rec = {"n_test": int(Xte.shape[0])}
            for nm, f in maps.items():
                mean, lo, hi = ci(p1_correct(f(Xte), Yte), rng)
                rec[nm] = {"p1": mean, "ci95": [lo, hi]}
            rows[label] = rec
            print(f"{label}: ortho {rec['ortho']['p1']:.3f} | linear-ls {rec['linear-ls']['p1']:.3f} "
                  f"| identity {rec['identity']['p1']:.3f}")
        except Exception as e:
            print(f"{label}: SKIPPED ({type(e).__name__}: {str(e)[:70]})")
    out = ROOT / "results" / "tatoeba"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "summary.json", "w"), indent=2)
    if rows:
        ao = np.mean([r["ortho"]["p1"] for r in rows.values()])
        al = np.mean([r["linear-ls"]["p1"] for r in rows.values()])
        print(f"\nAVG  ortho {ao:.3f} | linear-ls {al:.3f}  (n={len(rows)})")
    print("saved results/tatoeba/summary.json")


if __name__ == "__main__":
    main()
