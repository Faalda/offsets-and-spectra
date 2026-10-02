"""
Exp 2 (notes.txt): independent-SOURCE-FAMILY generalisation of mean dominance.

Question: does mean dominance predict the COST OF AN OFFSET across genuinely
independent source representations, rather than mainly distinguishing a small
number of repeated source families?  This addresses the paper's most
consequential evidential limitation: H1 has only four distinct held-out source
mean-dominance values.

Design (pre-registered as a NEW prospective batch, distinct from the 3 original):
  Source families (English FLORES), chosen for spread of mean dominance AND for
  being architecturally independent (not just new targets on an old source):
      mpnet     paraphrase-multilingual-mpnet   mean_dom ~0.08  (held-out src)
      gte-qwen2 Alibaba-NLP/gte-Qwen2-1.5B      ~0.23  (decoder LLM, d=1536)
      bsl-unc   bert-base-uncased               ~0.58  (vanilla BERT)
      e5-base   intfloat/multilingual-e5-base   ~0.74  (dev src, sentence-tformer)
      e5-large  intfloat/multilingual-e5-large  ~0.73  (larger, d=1024)
  ModernBERT-large is added if its embeddings exist on disk.
  Each source is aligned to the SAME 17 cached monolingual BERT targets, so the
  target side is held fixed and only the source geometry varies.

Per gap we measure (on TEST, maps fitted on TRAIN):
  probe P@1        ortho rotation applied WITHOUT centring (means carried)
  centred P@1       ortho rotation with centring (the offset correction)
  cca P@1           CCA(val-tuned lambda)            (whitening, for context)
  OFFSET COST       = centred_P@1 - probe_P@1         (the H1 response variable)
  mean_dom_x        source mean dominance             (the H1 predictor)
  A                 unpaired spectral mismatch

Primary analysis (PRE-DECLARED, frozen before looking at any new source result):
  H1-family: across independent source families, Spearman(source mean_dom,
             mean offset cost) > 0 at the family level (one point per source =
             median offset cost over its 17 targets).  Report the per-gap
             Spearman too, and a leave-one-source-family-out check.

Why family-level is primary: 17 targets share a source, so per-gap points are
not independent replications of the source effect; the family-level correlation
is the scientifically independent test (as notes.txt specifies).

Usage:  python experiments/review/source_families.py [cuda:0]
Writes: results/review/source_families.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import XL  # noqa: E402
from decomposition import mean_dom  # noqa: E402

CACHE = ROOT / "results" / "emb_cache"
R = ROOT / "results" / "review"

# source families: (label, encoder tag, notes). mean_dom ~ from prior inspection.
SOURCES = [
    ("mpnet", "sentence-transformers_paraphrase-multilingual-mpnet-base-v2", "held-out source"),
    ("gte-qwen2", "Alibaba-NLP_gte-Qwen2-1.5B-instruct", "decoder LLM, d=1536"),
    ("bert-unc", "bert-base-uncased", "vanilla BERT, d=768"),
    ("modernbert-large", "answerdotai_ModernBERT-large", "ModernBERT, d=768"),
    ("e5-base", "intfloat_multilingual-e5-base", "dev source, d=768"),
    ("e5-large", "intfloat_multilingual-e5-large", "larger, d=1024"),
]


def ld(tag, lang, split):
    return torch.load(CACHE / f"{tag}__{lang}__{split}.pt").float().cpu()


def pca768(Xtr, Xte):
    """Mean-preserving projection to 768 fitted on TRAIN source only.

    Uses the top-768 right singular vectors of the UNCENTERED X, so the mean
    direction (which dominates when mean dominance is high) is retained. This
    is required to test mean dominance across wide sources: standard centered
    PCA removes the mean by construction and would zero mean_dom_x for every
    non-768 source, making the diagnostic inapplicable by definition. The mean
    direction is the top singular vector precisely when mean dominance is
    large, so the projection preserves the construct being measured.
    """
    _, _, Vt = torch.linalg.svd(Xtr, full_matrices=False)
    P = Vt[:768].T
    return Xtr @ P, Xte @ P


def eval_gap(Xtr, Ytr, Xte, Yte):
    rot = C.ortho_map(Xtr, Ytr)
    probe = C.evalmap(rot, Xte, Yte)                       # rotation, no centring
    centred = C.evalmap(H.centred_map(rot, Xtr, Ytr), Xte, Yte)
    ccav, _, _ = H.cca_val_select(Xtr, Ytr)
    cca = C.evalmap(ccav, Xte, Yte)
    h = Xtr.shape[0] // 2
    return {"probe": probe, "centred": centred, "cca": cca,
            "offset_cost": centred - probe,
            "mean_dom_x": mean_dom(Xtr), "mean_dom_y": mean_dom(Ytr),
            "A": C.anisotropy(Xtr[:h], Ytr[h:2 * h])}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for sname, stag, note in SOURCES:
        try:
            Xtr_all, Xte_all = ld(stag, "eng_Latn", "dev"), ld(stag, "eng_Latn", "devtest")
        except FileNotFoundError:
            print(f"{sname}: embeddings not cached, skipping", flush=True); continue
        native_d = Xtr_all.shape[1]
        res[sname] = {"encoder": stag, "note": note, "native_d": native_d, "targets": {}}
        for tgt, enc, lab in XL:
            try:
                Ytr, Yte = ld(enc.replace("/", "_"), tgt, "dev"), ld(enc.replace("/", "_"), tgt, "devtest")
            except FileNotFoundError:
                continue
            if native_d != 768:
                Xtr, Xte = pca768(Xtr_all, Xte_all)
            else:
                Xtr, Xte = Xtr_all, Xte_all
            g = eval_gap(Xtr, Ytr, Xte, Yte)
            res[sname]["targets"][lab] = g
            print(f"{sname:<16}{lab:<4} md={g['mean_dom_x']:.3f} A={g['A']:.3f} "
                  f"probe={g['probe']:.3f} cent={g['centred']:.3f} cca={g['cca']:.3f} "
                  f"offset_cost={g['offset_cost']:+.4f}", flush=True)

    # ---- analysis ----
    families = {}
    for sname, s in res.items():
        oc = [t["offset_cost"] for t in s["targets"].values()]
        md = [t["mean_dom_x"] for t in s["targets"].values()]
        families[sname] = {"mean_dom_x": float(np.mean(md)),
                           "median_offset_cost": float(np.median(oc)),
                           "mean_offset_cost": float(np.mean(oc)),
                           "n_targets": len(oc)}

    # primary: family-level Spearman (source mean_dom vs median offset cost)
    fam_names = list(families.keys())
    md_fam = np.array([families[f]["mean_dom_x"] for f in fam_names])
    oc_fam = np.array([families[f]["median_offset_cost"] for f in fam_names])
    fam_rho, fam_p = spearmanr(md_fam, oc_fam)

    # secondary: per-gap Spearman pooled across all source-target gaps
    all_md, all_oc = [], []
    for s in res.values():
        for t in s["targets"].values():
            all_md.append(t["mean_dom_x"]); all_oc.append(t["offset_cost"])
    all_md, all_oc = np.array(all_md), np.array(all_oc)
    gap_rho, gap_p = spearmanr(all_md, all_oc)

    # leave-one-source-family-out at the family level
    lofo = {}
    for i, f in enumerate(fam_names):
        m = np.ones(len(fam_names), bool); m[i] = False
        if np.std(md_fam[m]) > 0 and np.std(oc_fam[m]) > 0:
            r, p = spearmanr(md_fam[m], oc_fam[m]); lofo[f] = {"rho": float(r), "p": float(p), "n": int(m.sum())}

    res["_analysis"] = {
        "primary_family_level": {"n_sources": len(fam_names),
                                 "spearman": float(fam_rho), "p": float(fam_p),
                                 "sources": {f: families[f] for f in fam_names}},
        "secondary_pooled_gap_level": {"n_gaps": len(all_md), "spearman": float(gap_rho), "p": float(gap_p)},
        "leave_one_source_family_out": lofo,
        "predeclared": "primary = family-level Spearman(source mean_dom, median offset cost); "
                       "secondary = pooled per-gap Spearman; LOO family sensitivity.",
    }
    out = R / "source_families.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    a = res["_analysis"]
    print("\n=== PRIMARY (family-level) ===")
    for f in fam_names:
        print(f"  {f:<16} md={families[f]['mean_dom_x']:.3f}  median_offset_cost={families[f]['median_offset_cost']:+.4f}")
    print(f"  Spearman(source_mean_dom, median_offset_cost) = {fam_rho:.3f}  p={fam_p:.4f}  (n={len(fam_names)})")
    print("\n=== SECONDARY (pooled per-gap) ===")
    print(f"  Spearman = {gap_rho:.3f}  p={gap_p:.4f}  (n={len(all_md)})")
    print("\n=== leave-one-source-family-out ===")
    for f, v in lofo.items():
        print(f"  drop {f:<16} rho={v['rho']:.3f} p={v['p']:.4f} n={v['n']}")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
