"""
Review-response benchmark: every gap in the paper x every map in common.py.

Answers: AI-W5 / R2-W3 (regularisation confound; tuned ridge, reduced-rank ridge,
whitened ridge, val-tuned CCA), R2-W3 / Q3 (regularised MLP incl. cross-modal),
R1-W4 (explicit protocol: all selection on a validation split of TRAIN),
R1-W3 / AI-minor (anisotropy computed from UNPAIRED disjoint halves).

Usage:  python experiments/review/benchmark.py GROUP [cuda:0]
        GROUP in {xling_e5, xling_labse, xarch, clip, xmodal, ctrl}
Writes results/review/benchmark_<GROUP>.json
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
from common import *  # noqa: E402,F401,F403
import common as C  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
XL = [("deu_Latn", "bert-base-german-cased", "de"), ("rus_Cyrl", "DeepPavlov/rubert-base-cased", "ru"),
      ("arb_Arab", "CAMeL-Lab/bert-base-arabic-camelbert-da", "ar"), ("zho_Hans", "bert-base-chinese", "zh"),
      ("fra_Latn", "camembert-base", "fr"), ("spa_Latn", "dccuchile/bert-base-spanish-wwm-cased", "es"),
      ("ita_Latn", "dbmdz/bert-base-italian-cased", "it"), ("por_Latn", "neuralmind/bert-base-portuguese-cased", "pt"),
      ("nld_Latn", "GroNLP/bert-base-dutch-cased", "nl"), ("tur_Latn", "dbmdz/bert-base-turkish-cased", "tr"),
      ("fin_Latn", "TurkuNLP/bert-base-finnish-cased-v1", "fi"), ("pes_Arab", "HooshvareLab/bert-base-parsbert-uncased", "fa"),
      ("jpn_Jpan", "cl-tohoku/bert-base-japanese-v3", "ja"), ("kor_Hang", "klue/bert-base", "ko"),
      ("pol_Latn", "dkleczek/bert-base-polish-cased-v1", "pl"), ("swe_Latn", "KB/bert-base-swedish-cased", "sv"),
      ("ind_Latn", "cahya/bert-base-indonesian-1.5G", "id")]
XA = ["bert-base-uncased", "roberta-base", "microsoft/deberta-base", "google/electra-base-discriminator"]


def gaps(group):
    if group in ("xling_e5", "xling_labse"):
        src = "intfloat/multilingual-e5-base" if group == "xling_e5" else "sentence-transformers/LaBSE"
        for tgt, enc, lab in XL:
            d = load_crosslingual("eng_Latn", tgt, src, "cpu", encoder_tgt=enc)
            yield f"{group.split('_')[1]}/{lab}", "cross-lingual", d.X_train, d.Y_train, d.X_test, d.Y_test
    elif group == "xarch":
        for a, b in list(itertools.combinations(XA, 2)) + [("intfloat/multilingual-e5-base", "sentence-transformers/LaBSE")]:
            d = load_crosslingual("eng_Latn", "eng_Latn", a, "cpu", encoder_tgt=b)
            yield f"xa/{a.split('/')[-1][:7]}->{b.split('/')[-1][:7]}", "cross-arch", d.X_train, d.Y_train, d.X_test, d.Y_test
    elif group == "ctrl":
        d = load_crosslingual("eng_Latn", "deu_Latn", "intfloat/multilingual-e5-base", "cpu",
                              encoder_tgt="bert-base-german-cased")
        g = torch.Generator().manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(768, 768, generator=g))
        E = lambda Z: 0.3 * Z.norm(dim=1, keepdim=True).mean() / 768 ** .5 * torch.randn(Z.shape, generator=g)
        yield "ctrl/rot+noise", "isotropic", d.X_train, d.X_train @ Q + E(d.X_train), d.X_test, d.X_test @ Q + E(d.X_test)
    elif group == "clip":
        for nm in ("mscoco", "flickr30k", "flickr8k"):
            t = torch.load(CACHE / f"clip_{nm}.pt"); X, Y = t["img"].float(), t["txt"].float()
            n = X.shape[0]; ntr = min(1000, int(n * 0.67))
            yield f"clip/{nm}", "jointly-trained", X[:ntr], Y[:ntr], X[ntr:ntr + 500], Y[ntr:ntr + 500]
    elif group == "xmodal":
        Xa = torch.load(CACHE / "librispeech__speech__openai_whisper-large-v3__test.pt").float()
        Ya = torch.load(CACHE / "librispeech__text__xlm-roberta-large__test.pt").float()
        for s in range(5):  # 5 random 80/20 splits (as in the paper)
            perm = torch.randperm(Xa.shape[0], generator=torch.Generator().manual_seed(s))
            ntr = int(0.8 * Xa.shape[0]); tr, te = perm[:ntr], perm[ntr:]
            yield f"xm/libri-whisper->xlmr/s{s}", "cross-modal", Xa[tr], Ya[tr], Xa[te], Ya[te]
        yield ("xm/coco-dinov2->bert", "cross-modal",
               torch.load(CACHE / "dinov2-base__coco_train_imgs.pt").float(),
               torch.load(CACHE / "bert-base-uncased__coco_train_caps.pt").float(),
               torch.load(CACHE / "dinov2-base__coco_test_imgs.pt").float(),
               torch.load(CACHE / "bert-base-uncased__coco_test_caps.pt").float())


def run_gap(Xtr, Ytr, Xte, Yte, device):
    T = C.Timer(); out, hp = {}, {}
    same = Xtr.shape[1] == Ytr.shape[1]
    closed = {"ortho": C.ortho_map, "cca": C.cca_map, "linear-ls": C.linls_map,
              "ortho_wmetric": C.ortho_wmetric_map}
    if same:
        closed["identity"] = C.identity_map
        closed["lowrank-r8"] = lambda X, Y: C.lowrank_closed(X, Y, 8)
    for k, f in closed.items():
        m = T(k, f, Xtr, Ytr); out[k] = C.evalmap(m, Xte, Yte)
    for k, f in {"cca-tuned": C.cca_tuned, "ridge": C.ridge_tuned, "rrr": C.rrr_tuned,
                 "wridge": C.wridge_tuned}.items():
        m, best, _ = T(k, f, Xtr, Ytr); out[k] = C.evalmap(m, Xte, Yte); hp[k] = best
    for k, kw in {"mlp-cos": dict(loss="cosine"), "mlp-nce": dict(loss="infonce"),
                  "linear-nce": dict(loss="infonce", arch="linear")}.items():
        m, best, _ = T(k, C.mlp_tuned, Xtr, Ytr, device=device, **kw)
        out[k] = C.evalmap(m, Xte, Yte); hp[k] = best
    # diagnostics
    h = Xtr.shape[0] // 2
    diag = {"aniso_paired": C.anisotropy(Xtr, Ytr),
            "aniso_paired_half": C.anisotropy(Xtr[:h], Ytr[:h]),
            "aniso_unpaired": C.anisotropy(Xtr[:h], Ytr[h:2 * h]),   # disjoint sentences, same n
            "self_aniso_x": C.self_anisotropy(Xtr), "self_aniso_y": C.self_anisotropy(Ytr),
            "n_train": Xtr.shape[0], "n_test": Xte.shape[0], "dx": Xtr.shape[1], "dy": Ytr.shape[1]}
    return {"p1": out, "hp": {k: str(v) for k, v in hp.items()}, "time_s": T.t, "diag": diag}


def main():
    group = sys.argv[1]; device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    outp = ROOT / "results" / "review" / f"benchmark_{group}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(outp.read_text()) if outp.exists() else {}
    for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
        if name in res:
            continue
        r = run_gap(Xtr.float(), Ytr.float(), Xte.float(), Yte.float(), device); r["regime"] = regime
        res[name] = r
        outp.write_text(json.dumps(res, indent=1))
        p = r["p1"]
        print(f"{name:<28} " + " ".join(f"{k}={v:.3f}" for k, v in p.items()), flush=True)
    print("saved", outp)


if __name__ == "__main__":
    main()
