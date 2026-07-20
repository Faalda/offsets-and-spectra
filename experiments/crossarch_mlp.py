"""
Add MLP retrieval to cross-arch results.
Reads existing results/crossarch/summary.json, runs MLP on each pair,
updates the json in-place with an 'mlp' key.
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
    ("bert-base-uncased",              "roberta-base",                       "BERT-RoBERTa"),
    ("bert-base-uncased",              "microsoft/deberta-base",             "BERT-DeBERTa"),
    ("roberta-base",                   "microsoft/deberta-base",             "RoBERTa-DeBERTa"),
    ("bert-base-uncased",              "google/electra-base-discriminator",  "BERT-ELECTRA"),
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
    out_path = ROOT / "results" / "crossarch" / "summary.json"
    rows = json.load(open(out_path))
    label_to_row = {r["pair"]: r for r in rows}

    print(f"{'pair':<20} {'mlp':>8}")
    for ea, eb, lab in PAIRS:
        try:
            d = load_crosslingual("eng_Latn", "eng_Latn", ea, "cpu", encoder_tgt=eb)
            Xtr, Ytr, Xte, Yte = d.X_train, d.Y_train, d.X_test, d.Y_test
            mlp_score = run_mlp(Xtr, Ytr, Xte, Yte)
            print(f"{lab:<20} {mlp_score:>8.3f}")
            if lab in label_to_row:
                label_to_row[lab]["mlp"] = mlp_score
            else:
                print(f"  WARNING: {lab} not found in existing rows")
        except Exception as ex:
            print(f"{lab:<20} FAILED ({type(ex).__name__}: {str(ex)[:80]})")

    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nUpdated {out_path}")


if __name__ == "__main__":
    main()
