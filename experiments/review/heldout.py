"""
Held-out evaluation of the pre-registered hypotheses (results/review/prereg.json).
No design choice below was tuned on these gaps.

Gaps:
  mpnet/<lang>  paraphrase-multilingual-mpnet-base-v2 (new source family) -> 17 cached BERTs
  e5/<lang>     e5-base -> NEW monolingual BERTs (languages never used in development)
  slurp/*       SLURP-TN speech: whisper->wav2vec2, whisper->XLM-R, wav2vec2->XLM-R
                (text targets de-duplicated: repeated transcripts make retrieval ambiguous)
Per gap: identity, rotation, centred rotation, CCA(val lambda), CCA+CSLS, MLP-InfoNCE (tuned),
         + diagnostics (ID on train, mean dominance, unpaired A).
Usage: python experiments/review/heldout.py [cuda:0]
"""
from __future__ import annotations
import json, sys, ast
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import XL  # noqa: E402
from decomposition import mean_dom  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
NEW_LANGS = [("ell_Grek", "nlpaueb/bert-base-greek-uncased-v1", "el"),
             ("heb_Hebr", "onlplab/alephbert-base", "he"),
             ("ron_Latn", "dumitrescustefan/bert-base-romanian-cased-v1", "ro"),
             ("hun_Latn", "SZTAKI-HLT/hubert-base-cc", "hu"),
             ("dan_Latn", "Maltehb/danish-bert-botxo", "da"),
             ("ces_Latn", "ufal/robeczech-base", "cs"),
             ("ukr_Cyrl", "youscan/ukr-roberta-base", "uk"),
             ("cat_Latn", "PlanTL-GOB-ES/roberta-base-ca", "ca")]


def gaps(device):
    for tgt, enc, lab in XL:
        d = load_crosslingual("eng_Latn", tgt, "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                              device, encoder_tgt=enc)
        yield f"mpnet/{lab}", "cross-lingual", d.X_train, d.Y_train, d.X_test, d.Y_test
    for tgt, enc, lab in NEW_LANGS:
        try:
            d = load_crosslingual("eng_Latn", tgt, "intfloat/multilingual-e5-base", device, encoder_tgt=enc)
        except Exception as e:
            print(f"e5/{lab} SKIPPED ({type(e).__name__}: {str(e)[:100]})", flush=True); continue
        yield f"e5/{lab}", "cross-lingual", d.X_train, d.Y_train, d.X_test, d.Y_test
    ld = lambda kind, m, s: torch.load(CACHE / f"slurptn__{kind}__{m}__{s}.pt").float()
    W, V, T = "openai_whisper-large-v3", "jonatasgrosman_wav2vec2-large-xlsr-53-arabic", "xlm-roberta-large"
    def tr(kind, m):  # fit on train+validation, test on test
        return torch.cat([ld(kind, m, "train"), ld(kind, m, "validation")]), ld(kind, m, "test")
    def dedup(Xte, Yte):
        _, idx = torch.unique(Yte, dim=0, return_inverse=True)
        first = {}
        for i, g in enumerate(idx.tolist()):
            first.setdefault(g, i)
        keep = torch.tensor(sorted(first.values()))
        return Xte[keep], Yte[keep]
    (w_tr, w_te), (v_tr, v_te), (t_tr, t_te) = tr("speech", W), tr("speech", V), tr("text", T)
    yield "slurp/whisper->wav2vec2", "speech-speech", w_tr, v_tr, w_te, v_te
    yield ("slurp/whisper->xlmr", "cross-modal", w_tr, t_tr, *dedup(w_te, t_te))
    yield ("slurp/wav2vec2->xlmr", "cross-modal", v_tr, t_tr, *dedup(v_te, t_te))


def run(Xtr, Ytr, Xte, Yte, device):
    same = Xtr.shape[1] == Ytr.shape[1]
    rot = C.ortho_map(Xtr, Ytr); ccav, lv, _ = H.cca_val_select(Xtr, Ytr)
    mlp, hp, _ = C.mlp_tuned(Xtr, Ytr, loss="infonce", device=device)
    p = {"identity": C.evalmap(C.identity_map(Xtr, Ytr), Xte, Yte) if same else -1.0,
         "rotation": C.evalmap(rot, Xte, Yte),
         "rotation+centre": C.evalmap(H.centred_map(rot, Xtr, Ytr), Xte, Yte),
         "cca-val": C.evalmap(ccav, Xte, Yte), "cca-val+csls": H.csls_p1(ccav, Xte, Yte),
         "mlp-nce": C.evalmap(mlp, Xte, Yte)}
    a, b, va, vb = C.split_val(Xtr, Ytr)
    val = {"identity": C.evalmap(C.identity_map(a, b), va, vb) if same else -1.0,
           "rotation": C.evalmap(C.ortho_map(a, b), va, vb),
           "rotation+centre": C.evalmap(H.centred_map(C.ortho_map(a, b), a, b), va, vb),
           "cca-val": max(C.evalmap(C.cca_map(a, b, l), va, vb) for l in H.LAMBDA_GRID),
           "mlp-nce": hp["val_p1"]}
    h = Xtr.shape[0] // 2
    diag = {"ID_train": C.p1(Xtr, Ytr) if same else 0.0, "mean_dom_x": mean_dom(Xtr), "mean_dom_y": mean_dom(Ytr),
            "A": C.anisotropy(Xtr[:h], Ytr[h:2 * h]), "lam_val": lv, "n_train": Xtr.shape[0], "n_test": Xte.shape[0]}
    return {"p1": p, "val": val, "diag": diag}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    out = ROOT / "results" / "review" / "heldout.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for name, regime, Xtr, Ytr, Xte, Yte in gaps(device):
        if name in res:
            continue
        r = run(Xtr.float().cpu(), Ytr.float().cpu(), Xte.float().cpu(), Yte.float().cpu(), device)
        r["regime"] = regime; res[name] = r; out.write_text(json.dumps(res, indent=1))
        print(name, {k: round(v, 3) for k, v in r["p1"].items()}, {k: round(v, 3) for k, v in r["diag"].items()}, flush=True)


if __name__ == "__main__":
    main()
