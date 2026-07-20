"""
Modern cross-architecture alignment: LaBSE vs multilingual-e5-base.
Both are modern multilingual sentence encoders (768d), independently trained,
architecturally distinct (BERT-based dual encoder vs XLM-R contrastive).
Embeddings are already cached — no downloads needed.
Adds results to results/crossarch/summary.json.
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
    # (enc_src, enc_tgt, label)
    ("intfloat/multilingual-e5-base", "sentence-transformers/LaBSE", "e5-base→LaBSE"),
]


# ── alignment helpers (same as rest of codebase) ───────────────────────────────

def p1(M, Y):
    a, b = F.normalize(M.float(), dim=-1), F.normalize(Y.float(), dim=-1)
    return ((a @ b.T).argmax(1) == torch.arange(a.shape[0])).float().mean().item()


def run_ortho(Xtr, Ytr, Xte, Yte):
    Xc, Yc = Xtr - Xtr.mean(0), Ytr - Ytr.mean(0)
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return p1(Xte @ R, Yte)


def run_cca(Xtr, Ytr, Xte, Yte, lam=0.1):
    n = Xtr.shape[0]; mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    d = Xc.shape[1]; I = torch.eye(d)
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


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    out_path = ROOT / "results" / "crossarch" / "summary.json"
    rows = json.load(open(out_path))
    existing_labels = {r["pair"] for r in rows}

    print(f"{'pair':<24} {'aniso':>6} {'ortho':>7} {'cca':>7} {'linear':>7} {'mlp':>7}")
    new_rows = []
    for enc_src, enc_tgt, lab in PAIRS:
        try:
            d = load_crosslingual("eng_Latn", "eng_Latn", enc_src, "cpu",
                                  encoder_tgt=enc_tgt)
            Xtr, Ytr = d.X_train, d.Y_train
            Xte, Yte = d.X_test,  d.Y_test

            aniso  = anisotropy(Xtr, Ytr)
            ortho  = run_ortho(Xtr, Ytr, Xte, Yte)
            cca    = run_cca(Xtr, Ytr, Xte, Yte)
            lin_ls = run_linear_ls(Xtr, Ytr, Xte, Yte)
            mlp    = run_mlp(Xtr, Ytr, Xte, Yte)

            print(f"{lab:<24} {aniso:>6.2f} {ortho:>7.3f} {cca:>7.3f} {lin_ls:>7.3f} {mlp:>7.3f}")

            row = dict(pair=lab, regime="cross-arch-modern",
                       aniso=round(aniso, 3), ortho=round(ortho, 3),
                       cca=round(cca, 3), linear_ls=round(lin_ls, 3),
                       mlp=round(mlp, 3))
            new_rows.append(row)

        except Exception as ex:
            print(f"{lab:<24} FAILED ({type(ex).__name__}: {str(ex)[:80]})")

    # append only truly new rows
    for r in new_rows:
        if r["pair"] not in existing_labels:
            rows.append(r)
        else:
            for i, old in enumerate(rows):
                if old["pair"] == r["pair"]:
                    rows[i] = r

    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nUpdated {out_path}")


if __name__ == "__main__":
    main()
