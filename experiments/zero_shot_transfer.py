"""
Zero-shot cross-lingual transfer via the orthogonal controller.

Setup: train a classifier HEAD once on English (frozen e5 embeddings, labelled).
For each target language, embed its test set with a *monolingual* target BERT,
map it into the English space with the closed-form orthogonal rotation R fitted on
unlabelled parallel sentences (FLORES; reused from results/align/.../R.pt), and
apply the English head --- with NO target-language labels.

This is the regime where the rigid prior beats LoRA on a *task*: LoRA needs
labelled target data to adapt; our rotation needs none.

Reported per language:
  no-align     English head on raw target embeddings (different space -> fails)
  ortho (ours) English head on R-aligned target embeddings (zero target labels)
  supervised   head trained on target labels (the with-labels upper bound that
               LoRA-style adaptation would require)

Saves results/transfer/summary.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks.crosslingual import embed_sentences      # noqa: E402

SRC_ENC = "intfloat/multilingual-e5-base"
DEVICE = "cuda:0"
TRAIN_CAP = 4000          # cap per-language train for speed
# target lang: (massive code, monolingual encoder, align-pair label)
TARGETS = [
    ("de-DE", "bert-base-german-cased", "en-de"),
    ("es-ES", "dccuchile/bert-base-spanish-wwm-cased", "en-es"),
    ("fr-FR", "camembert-base", "en-fr"),
    ("it-IT", "dbmdz/bert-base-italian-cased", "en-it"),
    ("pt-PT", "neuralmind/bert-base-portuguese-cased", "en-pt"),
    ("nl-NL", "GroNLP/bert-base-dutch-cased", "en-nl"),
    ("tr-TR", "dbmdz/bert-base-turkish-cased", "en-tr"),
    ("fi-FI", "TurkuNLP/bert-base-finnish-cased-v1", "en-fi"),
    ("ru-RU", "DeepPavlov/rubert-base-cased", "en-ru"),
    ("ar-SA", "CAMeL-Lab/bert-base-arabic-camelbert-da", "en-ar"),
    ("zh-CN", "bert-base-chinese", "en-zh"),
    ("fa-IR", "HooshvareLab/bert-base-parsbert-uncased", "en-fa"),
]


def load_massive(lang, split):
    from datasets import load_dataset
    ds = load_dataset("AmazonScience/massive", lang, split=split, trust_remote_code=True)
    return list(ds["utt"]), np.array(ds["intent"])


def load_R(pair_label):
    d = ROOT / "results" / "align" / pair_label
    sub = next(p for p in d.iterdir() if p.is_dir())
    return torch.load(sub / "R.pt")          # R: en-space @ R -> target-space


def head_fit(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(X, y)
    return clf


def main():
    # English head (e5 source space)
    en_tr_txt, en_tr_y = load_massive("en-US", "train")
    en_tr_txt, en_tr_y = en_tr_txt[:TRAIN_CAP], en_tr_y[:TRAIN_CAP]
    Xen = embed_sentences(en_tr_txt, SRC_ENC, DEVICE).numpy()
    head = head_fit(Xen, en_tr_y)
    print(f"English head trained on {len(en_tr_y)} utts, {len(set(en_tr_y))} intents")

    rows = {}
    for code, enc, pair in TARGETS:
        try:
            te_txt, te_y = load_massive(code, "test")
            Yte = embed_sentences(te_txt, enc, DEVICE).numpy()
            R = load_R(pair).numpy()
            # map target -> English space with R^T (R: en->target, orthogonal)
            Yte_aligned = Yte @ R.T
            acc_noalign = (head.predict(Yte) == te_y).mean()
            acc_ortho = (head.predict(Yte_aligned) == te_y).mean()
            # supervised upper bound (needs target labels)
            tr_txt, tr_y = load_massive(code, "train")
            tr_txt, tr_y = tr_txt[:TRAIN_CAP], tr_y[:TRAIN_CAP]
            Ytr = embed_sentences(tr_txt, enc, DEVICE).numpy()
            acc_sup = (head_fit(Ytr, tr_y).predict(Yte) == te_y).mean()
            rows[pair] = {"no_align": float(acc_noalign), "ortho": float(acc_ortho),
                          "supervised": float(acc_sup), "n_test": len(te_y)}
            print(f"{pair}: no-align {acc_noalign:.3f} | ortho(ours) {acc_ortho:.3f} "
                  f"| supervised {acc_sup:.3f}")
        except Exception as e:
            print(f"{pair}: SKIPPED ({type(e).__name__}: {str(e)[:70]})")
    out = ROOT / "results" / "transfer"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "summary.json", "w"), indent=2)
    if rows:
        avg = {k: np.mean([r[k] for r in rows.values()]) for k in ("no_align", "ortho", "supervised")}
        print(f"\nAVG  no-align {avg['no_align']:.3f} | ortho(ours) {avg['ortho']:.3f} "
              f"| supervised {avg['supervised']:.3f}")
    print("saved results/transfer/summary.json")


if __name__ == "__main__":
    main()
