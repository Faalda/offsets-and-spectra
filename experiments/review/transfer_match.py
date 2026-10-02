"""
Batch 6: does a LABEL-FREE affine match of the mapped features to the English head's training
distribution improve zero-shot cross-lingual intent transfer (MASSIVE, 12 languages)?

Motivation (reviewer suggestion): retrieval is insensitive to some changes that a frozen linear
head is sensitive to, so match the mapped target features to the distribution the English head was
trained on.  Everything used for the matching is label-free and comes from FLORES (never from MASSIVE):

  base map  f   = centred Procrustes target->English (as in transfer.py, 'ortho-centred'), or the
                  validation-tuned ridge map ('ridge').
  matching      = CORAL: with mapped FLORES-dev target features F (mean mu_t, covariance S_t) and
                  English FLORES-dev features E (mean mu_s, covariance S_s),
                  z -> (z - mu_t) (S_t + r I)^(-1/2) (S_s + r I)^(1/2) + mu_s,
                  r = rho * mean eigenvalue of the respective covariance.  Primary rho = 0.01;
                  rho in {0.001, 0.1} are declared sensitivity checks.
The SAME English head (logistic regression on all 11,514 English MASSIVE training utterances) is used.

PRE-DECLARED CRITERIA (frozen before running; results/review/prereg_transfer.json):
  T1  (primary) mean accuracy gain of 'ortho-centred + CORAL (rho=0.01)' over 'ortho-centred', averaged
      over the 12 languages, is >= 0.02, with a 95% bootstrap interval over languages (4,000
      resamples) excluding zero.
  T2  report whether the matched map exceeds the best existing label-free map of Table 3
      (CCA with label-free top-k, mean 0.329) on the 12-language mean.
  T3  report FLORES retrieval P@1 of base and matched maps (matching changes the query side only).
Results are reported whatever they show.

Usage: python experiments/review/transfer_match.py [cuda:0]
Writes: results/review/transfer_match.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
from tasks import load_crosslingual  # noqa: E402
import common as C  # noqa: E402
from transfer import TARGETS, SRC, massive, head, acc  # noqa: E402

RHOS = [0.01, 0.001, 0.1]


def sqrtm(S, r, inv):
    w, V = torch.linalg.eigh(S)
    w = w.clamp(min=0) + r * w.clamp(min=0).mean()
    p = -0.5 if inv else 0.5
    return (V * w.pow(p)) @ V.T


def coral(F, E, rho):
    mt, ms = F.mean(0), E.mean(0)
    St = (F - mt).T @ (F - mt) / F.shape[0]
    Ss = (E - ms).T @ (E - ms) / E.shape[0]
    A = sqrtm(St, rho, True) @ sqrtm(Ss, rho, False)
    return lambda z: (z - mt) @ A + ms


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    en_tr, en_te = massive("en-US", "train", SRC, device), massive("en-US", "test", SRC, device)
    h = head(en_tr["X"], en_tr["y"])
    res = {"_english_in_language": acc(h, en_te["X"], en_te["y"])}
    for code, flores, enc, lab in TARGETS:
        te = massive(code, "test", enc, device)
        d = load_crosslingual("eng_Latn", flores, SRC, "cpu", encoder_tgt=enc)
        Ytr, Xtr, Yte_f, Xte_f = d.Y_train, d.X_train, d.Y_test, d.X_test
        mY, mX = Ytr.mean(0), Xtr.mean(0)
        fo, _ = C.ortho_map(Ytr, Xtr)
        bases = {"ortho-centred": (lambda y, fo=fo, mY=mY, mX=mX: fo(y - mY) + mX),
                 "ridge": C.ridge_tuned(Ytr, Xtr)[0][0]}
        r = {}
        for bn, f in bases.items():
            r[bn] = acc(h, f(te["X"]), te["y"])
            r[f"flores_p1_{bn}"] = C.p1(f(Yte_f), Xte_f)
            for rho in RHOS:
                g = coral(f(Ytr), Xtr, rho)
                r[f"{bn}+coral{rho}"] = acc(h, g(f(te["X"])), te["y"])
                r[f"flores_p1_{bn}+coral{rho}"] = C.p1(g(f(Yte_f)), Xte_f)
        res[lab] = r
        print(lab, {k: round(v, 3) for k, v in r.items() if not k.startswith("flores")}, flush=True)
    langs = [l for _, _, _, l in TARGETS]
    keys = [k for k in res[langs[0]]]
    res["_avg"] = {k: float(np.mean([res[l][k] for l in langs])) for k in keys}
    rng = np.random.default_rng(0)
    gains = np.array([res[l]["ortho-centred+coral0.01"] - res[l]["ortho-centred"] for l in langs])
    boots = np.array([gains[rng.integers(0, len(gains), len(gains))].mean() for _ in range(4000)])
    res["_T1"] = {"mean_gain": float(gains.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                  "per_language_gain": dict(zip(langs, gains.tolist())),
                  "supported": bool(gains.mean() >= 0.02 and np.percentile(boots, 2.5) > 0)}
    prev = json.load(open(ROOT / "results/review/transfer_v2.json"))["_avg"]["cca-k"]
    res["_T2"] = {"cca_k_label_free_avg": prev, "matched_avg": res["_avg"]["ortho-centred+coral0.01"],
                  "matched_exceeds_cca_k": bool(res["_avg"]["ortho-centred+coral0.01"] > prev)}
    (ROOT / "results/review/transfer_match.json").write_text(json.dumps(res, indent=1, default=float))
    print("\nAVG", {k: round(v, 3) for k, v in res["_avg"].items()})
    print("T1", json.dumps(res["_T1"], default=float)[:400]); print("T2", res["_T2"])


if __name__ == "__main__":
    main()
