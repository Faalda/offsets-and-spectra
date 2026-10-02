"""
Development study for the hub-aware extension, on the 51 EXISTING gap instances.
Everything decided here is frozen into results/review/prereg.json BEFORE any new gap is
embedded (see hub_prereg.py / hub_heldout.py).

Per gap:
  diagnostics   rotation hubness on a validation split (N1 and N10 skew), in-sample
                rotation hubness on the training pairs (N10) -- the latter needs no split
  maps (test)   rotation, CCA(0.1), CCA(val-P@1 lambda), CCA(hub-lambda, in-sample),
                CCA(hub-lambda, unpaired val clouds), + CSLS and centring on rotation / CCA
Usage: python experiments/review/hub_dev.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "review"))
import common as C  # noqa: E402
import hub_common as H  # noqa: E402
from benchmark import gaps  # noqa: E402


def run(Xtr, Ytr, Xte, Yte):
    a, b, va, vb = C.split_val(Xtr, Ytr)
    rot_split = C.ortho_map(a, b)
    rot = C.ortho_map(Xtr, Ytr)
    d = {"hub1_rot_val": H.map_hubness(rot_split, va, vb, k=1),
         "hub10_rot_val": H.map_hubness(rot_split, va, vb, k=10),
         "hub10_rot_in": H.map_hubness(rot, Xtr, Ytr, k=10)}
    cca01 = C.cca_map(Xtr, Ytr, 0.1)
    ccav, lv, _ = H.cca_val_select(Xtr, Ytr)
    ccah, lh, hs = H.cca_hub_select(Xtr, Ytr, "in")
    # unpaired variant: fit on 80%, select lambda by hubness on the other 20% used as two
    # unpaired clouds (targets shuffled -> pairing provably unused), refit on all train
    perm = torch.randperm(vb.shape[0], generator=torch.Generator().manual_seed(1))
    _, lu, _ = H.cca_hub_select(a, b, "unpaired", va, vb[perm])
    ccau = C.cca_map(Xtr, Ytr, lu)
    p = {"rotation": C.evalmap(rot, Xte, Yte), "cca0.1": C.evalmap(cca01, Xte, Yte),
         "cca-val": C.evalmap(ccav, Xte, Yte), "cca-hub-in": C.evalmap(ccah, Xte, Yte),
         "cca-hub-unpaired": C.evalmap(ccau, Xte, Yte),
         "rotation+csls": H.csls_p1(rot, Xte, Yte), "cca-val+csls": H.csls_p1(ccav, Xte, Yte),
         "rotation+centre": C.evalmap(H.centred_map(rot, Xtr, Ytr), Xte, Yte),
         "cca-val+centre": C.evalmap(H.centred_map(ccav, Xtr, Ytr), Xte, Yte)}
    lam_curve = {str(l): C.evalmap(C.cca_map(Xtr, Ytr, l), Xte, Yte) for l in H.LAMBDA_GRID}
    return {"p1": p, "diag": d, "lam": {"val": lv, "hub_in": lh, "hub_unpaired": lu},
            "hub_curve_in": {str(k): v for k, v in hs.items()}, "test_curve": lam_curve}


def main():
    out = ROOT / "results" / "review" / "hub_dev.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for group in ("ctrl", "xling_e5", "xling_labse", "xarch", "clip", "xmodal"):
        for name, regime, Xtr, Ytr, Xte, Yte in gaps(group):
            if name in res:
                continue
            r = run(Xtr.float(), Ytr.float(), Xte.float(), Yte.float()); r["regime"] = regime
            res[name] = r; out.write_text(json.dumps(res, indent=1))
            print(name, {k: round(v, 3) for k, v in r["p1"].items()}, r["lam"],
                  {k: round(v, 2) for k, v in r["diag"].items()}, flush=True)


if __name__ == "__main__":
    main()
