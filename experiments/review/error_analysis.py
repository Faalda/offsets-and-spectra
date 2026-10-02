"""
AI-sugg 4: error analysis in the non-saturated regimes (en->ar, en->it, en->fr).

Per test query, for ortho and cca: correct?, source length (words), retrieval
margin (top1 - top2 cosine), rank of the true target.  Mechanism check: hubness.
Whitening is known to reduce hubness (a few targets being the nearest neighbour
of many queries), which is a symptom of the anisotropy / representation
degeneration of contextual embeddings.  We report N1 k-occurrence skewness and
the share of queries captured by the top-1% hubs, for each map.

Usage: python experiments/review/error_analysis.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402

PAIRS = [("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "ar"),
         ("ita_Latn", "dbmdz/bert-base-italian-cased", "it"),
         ("fra_Latn", "camembert-base", "fr"), ("deu_Latn", "bert-base-german-cased", "de")]


def stats(fx, fy, X, Y):
    a, b = F.normalize(fx(X), dim=-1), F.normalize(fy(Y), dim=-1)
    S = a @ b.T
    top2 = S.topk(2, dim=1)
    nn = top2.indices[:, 0]
    true = S.diag()
    rank = (S > true[:, None]).sum(1)
    k_occ = torch.bincount(nn, minlength=S.shape[1]).float()
    hubs = k_occ.sort(descending=True).values
    return {"correct": (nn == torch.arange(S.shape[0])).numpy(),
            "margin": (top2.values[:, 0] - top2.values[:, 1]).numpy(), "rank": rank.numpy(),
            "hub_skew": float(skew(k_occ.numpy())), "top1pct_hub_share": float(hubs[:max(1, len(hubs) // 100)].sum() / len(nn)),
            "frac_never_retrieved": float((k_occ == 0).float().mean())}


def main():
    from datasets import load_dataset
    en = load_dataset("Muennighoff/flores200", "eng_Latn", split="devtest", trust_remote_code=True)["sentence"]
    L = np.array([len(s.split()) for s in en])
    q = np.quantile(L, [0.25, 0.5, 0.75])
    res = {}
    for tgt, enc, lab in PAIRS:
        d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", "cpu", encoder_tgt=enc)
        r = {}
        for name, m in {"ortho": C.ortho_map(d.X_train, d.Y_train), "cca": C.cca_map(d.X_train, d.Y_train),
                        "ridge": C.ridge_tuned(d.X_train, d.Y_train)[0]}.items():
            s = stats(*m, d.X_test, d.Y_test)
            bins = np.digitize(L, q)
            r[name] = {"p1": float(s["correct"].mean()), "hub_skew": s["hub_skew"],
                       "top1pct_hub_share": s["top1pct_hub_share"], "frac_never_retrieved": s["frac_never_retrieved"],
                       "p1_by_len_quartile": [float(s["correct"][bins == i].mean()) for i in range(4)],
                       "median_rank_of_errors": float(np.median(s["rank"][~s["correct"].astype(bool)])) if (~s["correct"].astype(bool)).any() else 0.0}
            r[name]["_c"] = s["correct"]
        oc, cc = r["ortho"].pop("_c").astype(bool), r["cca"].pop("_c").astype(bool); r["ridge"].pop("_c")
        r["fixed_by_cca"] = int((cc & ~oc).sum()); r["broken_by_cca"] = int((oc & ~cc).sum())
        r["len_quartile_edges"] = q.tolist()
        res[lab] = r
        print(lab, json.dumps(r)[:900], flush=True)
    out = ROOT / "results" / "review" / "error_analysis.json"
    out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
