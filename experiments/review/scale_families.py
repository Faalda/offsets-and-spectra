"""
Batch 4: 20 NEW source encoders x 17 targets (configuration frozen in scale_config.py).

Per gap (maps fitted on FLORES dev, evaluated on devtest, same code as source_families.py):
  probe P@1 (rotation without centring), centred Procrustes, CCA (validation lambda),
  offset cost = centred - probe, whitening gain = CCA - centred, A (unpaired halves), md_y.

PRE-DECLARED ANALYSES (frozen before any new embedding was computed; see prereg_families.json):
  P1  source level: Spearman(native source mean dominance, median offset cost over the 17 targets)
      > 0 across the NEW sources; supported iff one-sided Monte Carlo permutation p < 0.05.
  P2  variance components of offset cost (balanced crossed source x target, residual = interaction +
      noise); supported iff the source share exceeds the target share at the point estimate AND in
      >= 90% of pigeonhole-bootstrap replicates.
  P3  within-source Spearman(A, whitening gain) (source-demeaned); supported iff the pigeonhole 95%
      bootstrap interval excludes 0 from below.
  P4  within-source Spearman(target mean dominance, offset cost); reported only (expected null).
  Secondary: the same P1-P3 on the pooled 26 sources (adding the six of Appendix A.2).
All results are reported whatever they show.

Usage: python experiments/review/scale_families.py            (full run)
       python experiments/review/scale_families.py --smoke     (code check on an OLD source)
Writes: results/review/scale_families.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import XL  # noqa: E402
from decomposition import mean_dom  # noqa: E402
import scale_config as CFG  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
R = ROOT / "results" / "review"


def ld(tag, lang, split):
    return torch.load(CACHE / f"{tag}__{lang}__{split}.pt").float().cpu()


def project(Xtr, Xte):
    """d>768: mean-preserving projection (uncentred SVD); otherwise native dimension."""
    if Xtr.shape[1] <= 768:
        return Xtr, Xte
    _, _, Vt = torch.linalg.svd(Xtr, full_matrices=False)
    P = Vt[:768].T
    return Xtr @ P, Xte @ P


def eval_gap(Xtr, Ytr, Xte, Yte):
    rot = C.ortho_map(Xtr, Ytr)
    probe = C.evalmap(rot, Xte, Yte)
    centred = C.evalmap(H.centred_map(rot, Xtr, Ytr), Xte, Yte)
    ccav, _, _ = H.cca_val_select(Xtr, Ytr)
    cca = C.evalmap(ccav, Xte, Yte)
    h = Xtr.shape[0] // 2
    return {"probe": probe, "centred": centred, "cca": cca,
            "offset_cost": centred - probe, "whiten_gain": cca - centred,
            "mean_dom_y": mean_dom(Ytr), "A": C.anisotropy(Xtr[:h], Ytr[h:2 * h])}


def run_source(tag, targets=None):
    Xtr_all, Xte_all = ld(tag, "eng_Latn", "dev"), ld(tag, "eng_Latn", "devtest")
    md_native = float(mean_dom(Xtr_all))
    Xtr, Xte = project(Xtr_all, Xte_all)
    out = {"tag": tag, "native_d": int(Xtr_all.shape[1]), "used_d": int(Xtr.shape[1]),
           "mean_dom_native": md_native, "targets": {}}
    for tgt, enc, lab in XL:
        if targets and lab not in targets:
            continue
        Ytr, Yte = ld(enc.replace("/", "_"), tgt, "dev"), ld(enc.replace("/", "_"), tgt, "devtest")
        out["targets"][lab] = eval_gap(Xtr, Ytr, Xte, Yte)
    return out


# ---------------------------------------------------------------- statistics
def components(Y):
    S, T = Y.shape
    gm, rm, cm = Y.mean(), Y.mean(1), Y.mean(0)
    ms_s = T * ((rm - gm) ** 2).sum() / (S - 1)
    ms_t = S * ((cm - gm) ** 2).sum() / (T - 1)
    ms_e = ((Y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((S - 1) * (T - 1))
    v = np.array([max((ms_s - ms_e) / T, 0.0), max((ms_t - ms_e) / S, 0.0), ms_e])
    return v / v.sum()


def within_rho(X, Y):
    Xd, Yd = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    return float(spearmanr(Xd.ravel(), Yd.ravel())[0])


def pigeon_idx(S, T, rng):
    return np.ix_(rng.integers(0, S, S), rng.integers(0, T, T))


def analyse(M, md, label):
    """M: dict of S x T matrices (offset_cost, whiten_gain, A, md_y); md: source mean dominance."""
    S, T = M["offset_cost"].shape
    rng = np.random.default_rng(CFG.SEED)
    res = {"label": label, "n_sources": S, "n_targets": T}
    # P1
    oc = np.median(M["offset_cost"], axis=1)
    rho = float(spearmanr(md, oc)[0]); p_sc = float(spearmanr(md, oc, alternative="greater")[1])
    rx, ry = rankdata(md), rankdata(oc)
    perm = np.array([rng.permutation(ry) for _ in range(20000)]) if S > 12 else None
    rx_c = rx - rx.mean(); py = perm - perm.mean(1, keepdims=True)
    perm_rho = (py @ rx_c) / (np.sqrt((py ** 2).sum(1)) * np.sqrt((rx_c ** 2).sum()))
    p_perm = float((1 + np.sum(perm_rho >= rho - 1e-12)) / (len(perm_rho) + 1))
    res["P1"] = {"rho": rho, "p_scipy_one_sided": p_sc, "p_perm_one_sided": p_perm,
                 "supported": bool(p_perm < 0.05)}
    # P2
    sh = components(M["offset_cost"])
    boots = np.array([components(M["offset_cost"][pigeon_idx(S, T, rng)]) for _ in range(CFG.N_BOOT)])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    frac = float(np.mean(boots[:, 0] > boots[:, 1]))
    res["P2"] = {"share_source_target_resid": sh.tolist(), "ci95_lo": lo.tolist(), "ci95_hi": hi.tolist(),
                 "frac_boot_source_gt_target": frac,
                 "supported": bool(sh[0] > sh[1] and frac >= 0.90)}
    shw = components(M["whiten_gain"])
    res["whiten_gain_shares"] = shw.tolist()
    # P3 / P4
    for key, x, y in [("P3", "A", "whiten_gain"), ("P4", "md_y", "offset_cost")]:
        r0 = within_rho(M[x], M[y])
        # resample rows/cols jointly for both matrices
        bs = []
        for _ in range(CFG.N_BOOT):
            ix = pigeon_idx(S, T, rng)
            bs.append(within_rho(M[x][ix], M[y][ix]))
        bs = np.array(bs); bs = bs[~np.isnan(bs)]
        ci = np.percentile(bs, [2.5, 97.5]).tolist()
        per = [float(spearmanr(M[x][i], M[y][i])[0]) for i in range(S)]
        res[key] = {"rho": r0, "ci95": ci, "n_sources_positive": int(sum(v > 0 for v in per)),
                    "per_source": per}
        if key == "P3":
            res[key]["supported"] = bool(ci[0] > 0)
    return res


def build_matrices(results):
    keys = list(results.keys())
    labs = sorted(set.intersection(*[set(results[k]["targets"]) for k in keys]))
    M = {n: np.array([[results[k]["targets"][t][n] for t in labs] for k in keys])
         for n in ("offset_cost", "whiten_gain", "A", "mean_dom_y")}
    M["md_y"] = M.pop("mean_dom_y")
    md = np.array([results[k]["mean_dom_native"] for k in keys])
    return keys, M, md


def main():
    if "--smoke" in sys.argv:
        old = json.load(open(R / "source_families.json"))["bert-unc"]["targets"]
        got = run_source("bert-base-uncased", targets={"de", "ru"})
        for lab in ("de", "ru"):
            for k in ("probe", "centred", "cca"):
                a, b = old[lab][k], got["targets"][lab][k]
                print(f"smoke bert-unc {lab} {k}: stored {a:.4f} recomputed {b:.4f} {'OK' if abs(a-b) < 1e-3 else 'MISMATCH'}")
        print("native md", got["mean_dom_native"], "used_d", got["used_d"]); return
    results, failed = {}, []
    for enc, note in CFG.NEW_SOURCES:
        tag = enc.replace("/", "_")
        try:
            results[enc] = run_source(tag)
            r = results[enc]; oc = np.median([t["offset_cost"] for t in r["targets"].values()])
            print(f"{enc:<45} d={r['native_d']:<5} md={r['mean_dom_native']:.3f} median_offset_cost={oc:+.4f}", flush=True)
        except Exception as e:
            failed.append((enc, f"{type(e).__name__}: {str(e)[:120]}")); print("FAILED", enc, failed[-1][1], flush=True)
    keys, M, md = build_matrices(results)
    out = {"sources": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, "failed": failed,
           "primary_new_sources": analyse(M, md, "new sources only")}
    # secondary: pooled with the six of Appendix A.2
    old = json.load(open(R / "source_families.json"))
    old_res = {}
    for s in CFG.OLD_SOURCES:
        tg = {lab: {"offset_cost": g["centred"] - g["probe"], "whiten_gain": g["cca"] - g["centred"],
                    "A": g["A"], "mean_dom_y": g["mean_dom_y"]} for lab, g in old[s]["targets"].items()}
        old_res[s] = {"targets": tg, "mean_dom_native": CFG.OLD_NATIVE_MD[s]}
    pooled = {**old_res, **results}
    pk, PM, pmd = build_matrices(pooled)
    out["secondary_pooled"] = analyse(PM, pmd, "pooled: new sources + six of A.2")
    out["md_and_offset_by_source"] = {k: {"md": float(results[k]["mean_dom_native"]),
        "median_offset_cost": float(np.median([t["offset_cost"] for t in results[k]["targets"].values()]))} for k in results}
    (R / "scale_families.json").write_text(json.dumps(out, indent=1, default=float))
    for name in ("primary_new_sources", "secondary_pooled"):
        a = out[name]; print("\n===", a["label"], f"(n={a['n_sources']} sources)")
        print(f"  P1 rho={a['P1']['rho']:+.3f} p_perm={a['P1']['p_perm_one_sided']:.4f} supported={a['P1']['supported']}")
        print(f"  P2 shares src/tgt/res={np.round(a['P2']['share_source_target_resid'],2)} frac(src>tgt)={a['P2']['frac_boot_source_gt_target']:.2f} supported={a['P2']['supported']}")
        print(f"  P3 rho={a['P3']['rho']:+.3f} CI={np.round(a['P3']['ci95'],3)} positive in {a['P3']['n_sources_positive']}/{a['n_sources']} supported={a['P3']['supported']}")
        print(f"  P4 rho={a['P4']['rho']:+.3f} CI={np.round(a['P4']['ci95'],3)}")
    print("failed:", failed)


if __name__ == "__main__":
    main()
