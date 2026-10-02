"""
Batch 5: do REAL low-ceiling gaps carry norm/position-conditional structure that a conditional
(piecewise) Procrustes map can exploit better than CCA?

Gaps (declared in advance): the 9 COCO image->text gaps and the 6 Laya image->text gaps
(3,000 train / 1,000 test pairs each), plus the ModernBERT base -> fine-tuned ModernBERT pair
(FLORES dev 997 train / devtest 1,012 test).  16 gaps.

Procedure per gap (all fitting on TRAIN only):
  1. Centred PCA to k=128 on each side (fitted on train).
  2. Reference maps in that 128-d space: centred Procrustes; CCA with validation lambda.
  3. Conditional candidates: piecewise centred Procrustes with G in {2,4} groups defined by quantile
     bins of the source's first principal coordinate ('pc1') or centred norm ('norm')
     (stress_common.DIAG_FAMILIES).  The label-free structure diagnostic D of each candidate is its
     gain in held-out MSE over global Procrustes on a 20% split of the training pairs
     (stress_common.cv_gain).  The candidate with maximal D is SELECTED; if max D <= 0 no structure
     is detected and the selected map is the global centred Procrustes.
  4. Test P@1 of: centred Procrustes, CCA(128-d), selected conditional map; full-dimension CCA on the
     raw 768-d spaces is reported as a reference.  Paired 95% bootstrap (2,000 resamples over test
     queries) for selected-minus-CCA.

PRE-DECLARED CRITERIA (frozen before running; see results/review/prereg_conditional.json):
  R1  (positive) on at least ONE gap the selected conditional map exceeds CCA(128-d) by >= 0.02 P@1
      with a paired 95% interval excluding 0.
  R2  (detector) report the number of gaps with max D > 0 and D's relation to the gain.
  CONTROLS: (a) the spiral twist gap (a=1, twist.py construction, known to defeat global linear maps)
      must yield a selected conditional map beating CCA by >= 0.02 with interval excluding 0
      (positive control); (b) an identity gap (Y = X) must yield max D <= 0.01 (negative control).
      If either control fails, the detector is not trusted and R1 is inconclusive.
Results are reported whatever they show.

Usage: python experiments/review/real_conditional.py
Writes: results/review/real_conditional.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
import stress_common as S  # noqa: E402

E, L, K = ROOT / "results/emb_cache", ROOT / "results/emb_cache/laya", ROOT / "results/emb_cache/coco4k"
KPC = 128
N_BOOT = 2000
rng = np.random.default_rng(0)


def pca_fit(Z, k=KPC):
    mu = Z.mean(0)
    _, _, Vt = torch.linalg.svd(Z - mu, full_matrices=False)
    P = Vt[:k].T
    return lambda z: (z - mu) @ P


def run_gap(Xtr, Ytr, Xte, Yte, full_cca=True):
    Xtr, Ytr, Xte, Yte = (t.float() for t in (Xtr, Ytr, Xte, Yte))
    out = {"n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}
    if full_cca:
        c, _, _ = H.cca_val_select(Xtr, Ytr)
        out["cca_full_dim"] = C.evalmap(c, Xte, Yte)
    fx, fy = pca_fit(Xtr), pca_fit(Ytr)
    Xtr, Ytr, Xte, Yte = fx(Xtr), fy(Ytr), fx(Xte), fy(Yte)
    ref = {"centred": S.global_procrustes(Xtr, Ytr)}
    ccav, _, _ = H.cca_val_select(Xtr, Ytr); ref["cca"] = ccav
    cand = {}
    for kind, G in S.DIAG_FAMILIES:
        D = S.cv_gain(Xtr, Ytr, kind, G, seed=0)
        cand[f"{kind}{G}"] = {"D": float(D), "kind": kind, "G": G}
    best = max(cand, key=lambda n: cand[n]["D"])
    out["candidates_D"] = {n: v["D"] for n, v in cand.items()}
    out["max_D"], out["selected"] = cand[best]["D"], best
    if cand[best]["D"] > 0:
        b = cand[best]
        sel = S.pw_map(Xtr, Ytr, S.make_assign(b["kind"], Xtr, b["G"]), b["G"])
    else:
        sel = ref["centred"]
        out["selected"] = "none (global)"
    v = {n: C.p1_vec(*(m[0](Xte), m[1](Yte))).numpy() for n, m in {**ref, "selected": sel}.items()}
    out["p1"] = {n: float(a.mean()) for n, a in v.items()}
    d = v["selected"] - v["cca"]; nq = len(d)
    boots = np.array([d[rng.integers(0, nq, nq)].mean() for _ in range(N_BOOT)])
    out["selected_minus_cca"] = {"mean": float(d.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist()}
    return out


def gaps():
    I = {f"dinov2": torch.load(K / "facebook_dinov2-base.pt"), "vit": torch.load(K / "google_vit-base-patch16-224-in21k.pt"),
         "convnext": torch.load(K / "facebook_convnext-base-224-22k.pt")}
    T = {t: torch.load(K / f"{m}.pt") for t, m in (("bert", "bert-base-uncased"), ("e5", "intfloat_e5-base-v2"),
                                                     ("mpnet", "sentence-transformers_all-mpnet-base-v2"))}
    for a in I:
        for b in T:
            yield f"coco/{a}->{b}", "image-text", I[a][:3000], T[b][:3000], I[a][3000:], T[b][3000:]
    LI = {"laya-tower": torch.load(L / "coco_laya_tower.pt"), "laya-connector": torch.load(L / "coco_laya_connector.pt")}
    for a in LI:
        for b in T:
            yield f"laya/{a}->{b}", "image-text", LI[a][:3000], T[b][:3000], LI[a][3000:], T[b][3000:]
    fl = lambda tag, s: torch.load(L / f"flores_{tag}_{s}.pt") if (L / f"flores_{tag}_{s}.pt").exists() else torch.load(E / f"{tag}__eng_Latn__{s}.pt")
    yield ("laya/modernbert-base->laya-modernbert", "fine-tune", fl("modernbert-base-large", "dev"), fl("laya-modernbert", "dev"),
           fl("modernbert-base-large", "devtest"), fl("laya-modernbert", "devtest"))


def controls():
    from twist import twist, norm_stats
    Ztr, Zte = S.load_base(); st = norm_stats(Ztr)
    yield "control/spiral a=1 (positive)", "control", Ztr, twist(Ztr, 1.0, st), Zte, twist(Zte, 1.0, st)
    yield "control/identity Y=X (negative)", "control", Ztr, Ztr.clone(), Zte, Zte.clone()


def main():
    res = {}
    for name, reg, Xtr, Ytr, Xte, Yte in list(controls()) + list(gaps()):
        r = run_gap(Xtr, Ytr, Xte, Yte, full_cca=not name.startswith("control"))
        r["regime"] = reg; res[name] = r
        m = r["selected_minus_cca"]
        print(f"{name:<44} maxD={r['max_D']:+.4f} sel={r['selected']:<12} centred={r['p1']['centred']:.3f} cca128={r['p1']['cca']:.3f} "
              f"sel={r['p1']['selected']:.3f} diff={m['mean']:+.4f} [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]", flush=True)
    real = {k: v for k, v in res.items() if v["regime"] != "control"}
    pos = [k for k, v in real.items() if v["selected_minus_cca"]["mean"] >= 0.02 and v["selected_minus_cca"]["ci95"][0] > 0]
    cpos = res["control/spiral a=1 (positive)"]["selected_minus_cca"]
    ctrl_ok = (cpos["mean"] >= 0.02 and cpos["ci95"][0] > 0) and res["control/identity Y=X (negative)"]["max_D"] <= 0.01
    summ = {"R1_gaps_positive": pos, "R1_supported": bool(len(pos) > 0), "controls_ok": bool(ctrl_ok),
            "R2_gaps_with_maxD_gt_0": [k for k, v in real.items() if v["max_D"] > 0], "n_real_gaps": len(real)}
    (ROOT / "results/review/real_conditional.json").write_text(json.dumps({"gaps": res, "summary": summ}, indent=1, default=float))
    print("\nSUMMARY", json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
