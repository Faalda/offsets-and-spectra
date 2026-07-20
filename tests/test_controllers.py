"""Sanity tests for the DSV controller family.

Run:  nadi/.venv/bin/python3 -m pytest DSV/tests/ -q
or:   nadi/.venv/bin/python3 DSV/tests/test_controllers.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import available_controllers, build_controller
from dsv.controllers import OrthoDSV, _cayley


def test_all_variants_shape():
    h = torch.randn(5, 16)
    for name in available_controllers():
        ctrl = build_controller(name, dim=16, n_layers=1)
        out = ctrl(h, layer=0)
        assert out.shape == h.shape, f"{name} changed shape"


def test_ortho_is_norm_preserving():
    # R is orthogonal, so ‖R(h+δ)‖ == ‖h+δ‖ exactly (the invariant is on the
    # shifted input, not on h, since δ may be nonzero).
    for param in ("cayley", "householder"):
        ctrl = build_controller("ortho", dim=24, n_layers=1, parametrization=param)
        h = torch.randn(7, 24)
        shifted = h + ctrl.delta[0].detach()
        out = ctrl(h)
        assert torch.allclose(out.norm(dim=-1), shifted.norm(dim=-1), atol=1e-4), \
            f"{param} not norm-preserving"


def test_cayley_identity_init():
    # raw = 0  ->  R = I
    R = _cayley(torch.zeros(8, 8))
    assert torch.allclose(R, torch.eye(8), atol=1e-5)


def test_cayley_is_orthogonal():
    raw = torch.randn(10, 10)
    R = _cayley(raw)
    assert torch.allclose(R @ R.T, torch.eye(10), atol=1e-4)


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    ctrl = build_controller("gated", dim=12, n_layers=2)
    h = torch.randn(3, 12)
    before = ctrl(h, layer=1)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ctrl.pt"
        ctrl.save(path)
        ctrl2 = build_controller("gated", dim=12, n_layers=2).load(path)
    assert torch.allclose(before, ctrl2(h, layer=1), atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
