"""
Zero-shot cross-lingual intent transfer, corrected + extended (AI-W6, AI-sugg 3).

Fixes a sampling bug in experiments/zero_shot_transfer.py: it took the FIRST
4000 MASSIVE train utterances, which cover only 42 of the 60 intents, capping
every head (incl. the "supervised" reference) at ~0.33.  Here heads are trained
on the FULL English train split (11,514 utts, 60 intents).

For every alignment map fitted on unlabelled FLORES pairs (target -> English),
the SAME English head is applied to mapped target test utterances.  CCA maps
both sides into the shared CCA space, so its English head is trained on
CCA-projected English features (still English labels only).

Also reports: English in-language accuracy (head ceiling) and the supervised
target head (full target train labels), plus FLORES retrieval P@1 of each map in
the same direction, to test whether the retrieval ranking predicts the transfer
ranking.

Usage: python experiments/review/transfer.py [cuda:0]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual                   # noqa: E402
from tasks.crosslingual import embed_sentences        # noqa: E402
import common as C                                    # noqa: E402

SRC = "intfloat/multilingual-e5-base"
TARGETS = [("de-DE", "deu_Latn", "bert-base-german-cased", "de"),
           ("es-ES", "spa_Latn", "dccuchile/bert-base-spanish-wwm-cased", "es"),
           ("fr-FR", "fra_Latn", "camembert-base", "fr"),
           ("it-IT", "ita_Latn", "dbmdz/bert-base-italian-cased", "it"),
           ("pt-PT", "por_Latn", "neuralmind/bert-base-portuguese-cased", "pt"),
           ("nl-NL", "nld_Latn", "GroNLP/bert-base-dutch-cased", "nl"),
           ("tr-TR", "tur_Latn", "dbmdz/bert-base-turkish-cased", "tr"),
           ("fi-FI", "fin_Latn", "TurkuNLP/bert-base-finnish-cased-v1", "fi"),
           ("ru-RU", "rus_Cyrl", "DeepPavlov/rubert-base-cased", "ru"),
           ("ar-SA", "arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "ar"),
           ("zh-CN", "zho_Hans", "bert-base-chinese", "zh"),
           ("fa-IR", "pes_Arab", "HooshvareLab/bert-base-parsbert-uncased", "fa")]
CACHE = ROOT / "results" / "emb_cache" / "massive"


def massive(lang, split, enc, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{enc.replace('/', '_')}__{lang}__{split}.pt"
    if p.exists():
        return torch.load(p)
    from datasets import load_dataset
    ds = load_dataset("AmazonScience/massive", lang, split=split, trust_remote_code=True)
    out = {"X": embed_sentences(list(ds["utt"]), enc, device), "y": torch.tensor(ds["intent"])}
    torch.save(out, p)
    return out


def cca_trunc(Y, X, k):
    fy, fx = C.cca_map(Y, X)
    return (lambda y: fy(y)[:, :k]), (lambda x: fx(x)[:, :k])


def head(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=10.0).fit(X.numpy(), y.numpy())


def acc(h, X, y):
    return float((h.predict(X.numpy()) == y.numpy()).mean())


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    en_tr, en_te = massive("en-US", "train", SRC, device), massive("en-US", "test", SRC, device)
    print(f"EN train {len(en_tr['y'])} utts, {len(set(en_tr['y'].tolist()))} intents", flush=True)
    h_raw = head(en_tr["X"], en_tr["y"])
    res = {"_english_in_language": acc(h_raw, en_te["X"], en_te["y"])}
    print("EN in-language acc", res["_english_in_language"], flush=True)
    for code, flores, enc, lab in TARGETS:
        tr, te = massive(code, "train", enc, device), massive(code, "test", enc, device)
        d = load_crosslingual("eng_Latn", flores, SRC, "cpu", encoder_tgt=enc)
        # maps fitted target -> English on unlabelled FLORES dev
        Ytr, Xtr, Yte_f, Xte_f = d.Y_train, d.X_train, d.Y_test, d.X_test
        mY, mX = Ytr.mean(0), Xtr.mean(0)
        fo, _ = C.ortho_map(Ytr, Xtr)
        # CCA truncated to k components, k chosen by label-free FLORES validation retrieval
        (cca_k, best_k, _) = C.tune(lambda a, b, k: cca_trunc(a, b, k), [64, 128, 256, 512, 768], Ytr, Xtr)
        maps = {"identity": C.identity_map(Ytr, Xtr), "ortho": C.ortho_map(Ytr, Xtr),
                "ortho-centred": (lambda y: fo(y - mY) + mX, lambda x: x), "cca-k": cca_k,
                "cca": C.cca_map(Ytr, Xtr), "ridge": C.ridge_tuned(Ytr, Xtr)[0],
                "mlp-nce": C.mlp_tuned(Ytr, Xtr, loss="infonce", device=device)[0]}
        r = {"supervised": acc(head(tr["X"], tr["y"]), te["X"], te["y"])}
        for k, (fy, fx) in maps.items():   # fy: target->shared, fx: English->shared
            hk = h_raw if not k.startswith("cca") else head(fx(en_tr["X"]), en_tr["y"])
            r[k] = acc(hk, fy(te["X"]), te["y"])
            r[f"flores_p1_{k}"] = C.p1(fy(Yte_f), fx(Xte_f))
        r["cca_k_selected"] = best_k
        res[lab] = r
        print(lab, {k: round(v, 3) for k, v in r.items()}, flush=True)
    keys = [k for k in res["de"] if k != "cca_k_selected"]
    res["_avg"] = {k: float(np.mean([res[l][k] for _, _, _, l in TARGETS])) for k in keys}
    print("AVG", {k: round(v, 3) for k, v in res["_avg"].items()})
    out = ROOT / "results" / "review" / "transfer_v2.json"
    out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()
