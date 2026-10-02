"""
Second pre-registered held-out batch (prereg2.json), part B: zero-shot sentiment transfer.
cardiffnlp/tweet_sentiment_multilingual (3 classes). English head (logistic regression) on frozen
e5-base embeddings of the English train split; applied to 6 target languages after mapping their
monolingual-BERT embeddings into the English space with maps fitted on unlabelled FLORES pairs.
Same maps and protocol as transfer.py (intent task).
Usage: python experiments/review/sentiment.py [cuda:0]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
from tasks.crosslingual import embed_sentences  # noqa: E402
import common as C  # noqa: E402
from transfer import cca_trunc, head, acc  # noqa: E402

SRC = "intfloat/multilingual-e5-base"
T = [("arabic", "arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "ar"),
     ("french", "fra_Latn", "camembert-base", "fr"), ("german", "deu_Latn", "bert-base-german-cased", "de"),
     ("italian", "ita_Latn", "dbmdz/bert-base-italian-cased", "it"),
     ("portuguese", "por_Latn", "neuralmind/bert-base-portuguese-cased", "pt"),
     ("spanish", "spa_Latn", "dccuchile/bert-base-spanish-wwm-cased", "es")]
CACHE = ROOT / "results" / "emb_cache" / "tweetsent"


def data(cfg, split, enc, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{enc.replace('/', '_')}__{cfg}__{split}.pt"
    if p.exists():
        return torch.load(p)
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", cfg, split=split)
    o = {"X": embed_sentences(list(ds["text"]), enc, device), "y": torch.tensor(ds["label"])}
    torch.save(o, p); return o


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    en_tr, en_te = data("english", "train", SRC, device), data("english", "test", SRC, device)
    h_raw = head(en_tr["X"], en_tr["y"])
    res = {"_english_in_language": acc(h_raw, en_te["X"], en_te["y"])}
    for cfg, flores, enc, lab in T:
        tr, te = data(cfg, "train", enc, device), data(cfg, "test", enc, device)
        d = load_crosslingual("eng_Latn", flores, SRC, "cpu", encoder_tgt=enc)
        Ytr, Xtr, Yf, Xf = d.Y_train, d.X_train, d.Y_test, d.X_test
        mY, mX = Ytr.mean(0), Xtr.mean(0); fo, _ = C.ortho_map(Ytr, Xtr)
        cca_k, bk, _ = C.tune(lambda a, b, k: cca_trunc(a, b, k), [64, 128, 256, 512, 768], Ytr, Xtr)
        maps = {"identity": C.identity_map(Ytr, Xtr), "ortho": C.ortho_map(Ytr, Xtr),
                "ortho-centred": (lambda y, fo=fo, mY=mY, mX=mX: fo(y - mY) + mX, lambda x: x),
                "cca": C.cca_map(Ytr, Xtr), "cca-k": cca_k, "ridge": C.ridge_tuned(Ytr, Xtr)[0],
                "mlp-nce": C.mlp_tuned(Ytr, Xtr, loss="infonce", device=device)[0]}
        r = {"supervised": acc(head(tr["X"], tr["y"]), te["X"], te["y"]), "cca_k": bk}
        for k, (fy, fx) in maps.items():
            hk = h_raw if not k.startswith("cca") else head(fx(en_tr["X"]), en_tr["y"])
            r[k] = acc(hk, fy(te["X"]), te["y"]); r[f"p1_{k}"] = C.p1(fy(Yf), fx(Xf))
        res[lab] = r; print(lab, {k: round(v, 3) for k, v in r.items()}, flush=True)
    labs = [t[3] for t in T]; M = ["identity", "ortho", "ortho-centred", "cca", "cca-k", "ridge", "mlp-nce"]
    res["_avg"] = {k: float(np.mean([res[l][k] for l in labs])) for k in M + ["supervised"] + [f"p1_{m}" for m in M]}
    fitted = [m for m in M if m != "identity"]
    res["_P5_spearman_p1_vs_acc"] = float(spearmanr([res["_avg"][f"p1_{m}"] for m in fitted], [res["_avg"][m] for m in fitted])[0])
    res["_best_map"] = max(fitted, key=lambda m: res["_avg"][m])
    print("AVG", {k: round(v, 3) for k, v in res["_avg"].items()}, "P5 rho", res["_P5_spearman_p1_vs_acc"], "best", res["_best_map"])
    (ROOT / "results" / "review" / "sentiment.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
