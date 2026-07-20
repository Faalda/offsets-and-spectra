"""
Run MLP on all 12 cross-lingual pairs and print the average.
All embeddings are cached — no model downloads needed.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks import load_crosslingual  # noqa: E402

torch.manual_seed(0)

PAIRS = [
    ("deu_Latn", "bert-base-german-cased",                        "en-de"),
    ("rus_Cyrl", "DeepPavlov/rubert-base-cased",                  "en-ru"),
    ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da",       "en-ar"),
    ("zho_Hans", "bert-base-chinese",                             "en-zh"),
    ("fra_Latn", "camembert-base",                                "en-fr"),
    ("spa_Latn", "dccuchile/bert-base-spanish-wwm-cased",         "en-es"),
    ("ita_Latn", "dbmdz/bert-base-italian-cased",                 "en-it"),
    ("por_Latn", "neuralmind/bert-base-portuguese-cased",         "en-pt"),
    ("nld_Latn", "GroNLP/bert-base-dutch-cased",                  "en-nl"),
    ("tur_Latn", "dbmdz/bert-base-turkish-cased",                 "en-tr"),
    ("fin_Latn", "TurkuNLP/bert-base-finnish-cased-v1",           "en-fi"),
    ("pes_Arab", "HooshvareLab/bert-base-parsbert-uncased",       "en-fa"),
]


def p1(M, Y):
    a, b = F.normalize(M, dim=-1), F.normalize(Y, dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


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


def main():
    scores = {}
    print(f"{'pair':<8} {'mlp':>8}")
    for tgt_lang, enc_tgt, lab in PAIRS:
        try:
            d = load_crosslingual("eng_Latn", tgt_lang,
                                  "intfloat/multilingual-e5-base", "cpu",
                                  encoder_tgt=enc_tgt)
            score = run_mlp(d.X_train, d.Y_train, d.X_test, d.Y_test)
            scores[lab] = score
            print(f"{lab:<8} {score:>8.3f}")
        except Exception as ex:
            print(f"{lab:<8} FAILED ({type(ex).__name__}: {str(ex)[:80]})")

    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\navg MLP over {len(scores)} pairs: {avg:.3f}")

    # Save
    out = ROOT / "results" / "crosslingual_mlp.json"
    json.dump(scores, open(out, "w"), indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
