"""
Domain-Shift Vectors (DSV) — a family of parameter-efficient controllers that
transform frozen representations.

All controllers share one interface:

    ctrl = build_controller("ortho", dim=64, n_layers=1)
    h_prime = ctrl(h, layer=0)          # h: (..., dim)  ->  (..., dim)

They are meant to be inserted on top of a *frozen* backbone. Only the
controller's own parameters are trained. The point of the package is to compare,
on equal footing, how much expressiveness each parametrization buys and at what
parameter cost.

Variants
--------
    dsv      h' = h + δ                         additive shift              (baseline)
    scaled   h' = h + s·δ           (s scalar)  ── absorbs into δ ──        (ablation)
    diag     h' = h + w⊙δ           (w vector)  ── absorbs into δ ──        (ablation)
    matrix   h' = h + Wδ            (W d×d)     ── absorbs into δ ──        (ablation)
    gated    h' = h + σ(h·a+b)·δ                input-dependent magnitude
    mixture  h' = h + Σ_k g_k(h)·δ_k            MoE of shift directions
    ortho    h' = R(h + δ),  R∈SO(d)            rigid alignment (rotation+shift)
    ogated   h' = R(h + σ(h·a)·δ)               ortho + gate
    lowrank  h' = h + B(Aᵀh)                    LoRA-on-activations baseline
    affine   h' = h + Wh + δ                    full linear adapter baseline

The `scaled`/`diag`/`matrix` variants are included on purpose: they look more
expressive than `dsv` but a fixed linear map applied to a learnable vector is
absorbed into that vector during training, so they cannot beat `dsv`. The
experiments are expected to confirm this empirically.

`ortho` is the novel contribution: an *orthogonal* transform of the activation
cannot be folded into δ (it depends on h) and is not a low-rank additive update
(so it is outside the LoRA / adapter family), yet it stays parameter-efficient.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REGISTRY: dict[str, type["DSVController"]] = {}


def register(name: str):
    def deco(cls):
        cls.variant_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def build_controller(name: str, dim: int, n_layers: int = 1, **kwargs) -> "DSVController":
    if name not in _REGISTRY:
        raise ValueError(f"Unknown controller '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](dim, n_layers, **kwargs)


def available_controllers() -> list[str]:
    return sorted(_REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class DSVController(nn.Module):
    """Base controller. Subclasses implement `transform(h, layer)`."""

    variant_name = "base"

    def __init__(self, dim: int, n_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, h: torch.Tensor, layer: int = 0) -> torch.Tensor:
        return self.transform(h, layer)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": self.variant_name, "dim": self.dim,
                    "n_layers": self.n_layers, "state": self.state_dict()}, path)

    def load(self, path: str | Path) -> "DSVController":
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["state"])
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Additive family
# ─────────────────────────────────────────────────────────────────────────────

@register("dsv")
class DSV(DSVController):
    """h' = h + δ_l."""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)

    def transform(self, h, layer=0):
        return h + self.delta[layer]


@register("scaled")
class ScaledDSV(DSVController):
    """h' = h + s_l·δ_l. (s·δ collapses to one vector — ablation.)"""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        self.scale = nn.Parameter(torch.ones(n_layers))

    def transform(self, h, layer=0):
        return h + self.scale[layer] * self.delta[layer]


@register("diag")
class DiagDSV(DSVController):
    """h' = h + w_l⊙δ_l. (elementwise w⊙δ collapses to one vector — ablation.)"""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        self.w = nn.Parameter(torch.ones(n_layers, dim))

    def transform(self, h, layer=0):
        return h + self.w[layer] * self.delta[layer]


@register("matrix")
class MatrixDSV(DSVController):
    """h' = h + W_l·δ_l. (Wδ collapses to one vector — ablation; should ≈ dsv.)"""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        self.W = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(n_layers, 1, 1))

    def transform(self, h, layer=0):
        return h + self.delta[layer] @ self.W[layer].T


# ─────────────────────────────────────────────────────────────────────────────
# Multiplicative gating family
# ─────────────────────────────────────────────────────────────────────────────

@register("chgate")
class ChannelGate(DSVController):
    """h' = σ(g_l) ⊙ h. Input-independent per-channel multiplicative gate
    (FiLM-style scale; channel-gating baseline). A fixed *diagonal* map on h:
    it can rescale (gate) channels in (0,1) but cannot translate or rotate, so it
    cannot close a shift or a rotation gap. Acts on h (not absorbed into a shift),
    but is the most restricted such controller. Params: d per layer."""

    def __init__(self, dim, n_layers=1, gate_init: float = 4.0):
        super().__init__(dim, n_layers)
        # σ(4) ≈ 0.98 → starts near identity
        self.g = nn.Parameter(torch.full((n_layers, dim), float(gate_init)))

    def transform(self, h, layer=0):
        return torch.sigmoid(self.g[layer]) * h


# ─────────────────────────────────────────────────────────────────────────────
# Input-dependent family
# ─────────────────────────────────────────────────────────────────────────────

@register("gated")
class GatedDSV(DSVController):
    """h' = h + σ(h·a_l + b_l)·δ_l. Shift magnitude varies per input."""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        self.a = nn.Parameter(torch.zeros(n_layers, dim))
        self.b = nn.Parameter(torch.zeros(n_layers))

    def transform(self, h, layer=0):
        gate = torch.sigmoid((h * self.a[layer]).sum(-1, keepdim=True) + self.b[layer])
        return h + gate * self.delta[layer]


@register("mlpgated")
class MLPGatedDSV(DSVController):
    """h' = h + σ(W₂·φ(W₁·h) + b) ⊙ δ_l.

    A bottleneck MLP predicts a *per-dimension* gate that selects which entries of
    the shift δ are active for this input. Unlike `gated` (a single scalar), the
    gate is a full d-vector; unlike LoRA, the bottleneck output gates a fixed shift
    rather than being added to a weight."""

    def __init__(self, dim, n_layers=1, bottleneck: int = 16, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.bottleneck = bottleneck
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        self.W1 = nn.Parameter(torch.randn(n_layers, dim, bottleneck) * (dim ** -0.5))
        self.W2 = nn.Parameter(torch.zeros(n_layers, bottleneck, dim))   # gate starts at σ(b)
        self.b = nn.Parameter(torch.zeros(n_layers, dim))

    def transform(self, h, layer=0):
        gate = torch.sigmoid(F.relu(h @ self.W1[layer]) @ self.W2[layer] + self.b[layer])
        return h + gate * self.delta[layer]


@register("adapter")
class Adapter(DSVController):
    """h' = h + W₂·φ(W₁·h + b₁) + b₂. A Houlsby-style bottleneck adapter with NO shift
    vector — the control baseline for `mlpgated`: a *free* residual transform at the same
    parameter count. If `mlpgated` beats this, the gain comes from gating a domain-shift
    vector, not merely from the bottleneck's capacity."""

    def __init__(self, dim, n_layers=1, bottleneck: int = 16):
        super().__init__(dim, n_layers)
        self.bottleneck = bottleneck
        self.W1 = nn.Parameter(torch.randn(n_layers, dim, bottleneck) * (dim ** -0.5))
        self.b1 = nn.Parameter(torch.zeros(n_layers, bottleneck))
        self.W2 = nn.Parameter(torch.zeros(n_layers, bottleneck, dim))   # init 0 → starts at identity
        self.b2 = nn.Parameter(torch.zeros(n_layers, dim))

    def transform(self, h, layer=0):
        return h + F.relu(h @ self.W1[layer] + self.b1[layer]) @ self.W2[layer] + self.b2[layer]


@register("mixture")
class MixtureDSV(DSVController):
    """h' = h + Σ_k softmax(h·G_l)_k · δ_l^k. MoE over K shift directions."""

    def __init__(self, dim, n_layers=1, n_experts: int = 4, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.n_experts = n_experts
        self.delta = nn.Parameter(torch.randn(n_layers, n_experts, dim) * init_scale)
        self.gate = nn.Parameter(torch.zeros(n_layers, dim, n_experts))

    def transform(self, h, layer=0):
        w = torch.softmax(h @ self.gate[layer], dim=-1)        # (..., K)
        shift = w @ self.delta[layer]                          # (..., d)
        return h + shift


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal family — the novel contribution
# ─────────────────────────────────────────────────────────────────────────────

def _cayley(raw: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained d×d matrix to SO(d) via the Cayley transform.

    A = (raw - rawᵀ)/2 is skew-symmetric; R = (I - A)(I + A)⁻¹ is orthogonal.
    raw = 0  ->  A = 0  ->  R = I  (exact identity initialisation)."""
    d = raw.shape[-1]
    A = (raw - raw.transpose(-1, -2)) * 0.5
    eye = torch.eye(d, device=raw.device, dtype=raw.dtype)
    return torch.linalg.solve(eye + A, eye - A)


def _householder(h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply a product of Householder reflections to h without materialising R.

    V: (K, d) reflection vectors. Each reflection H_v = I - 2 vvᵀ/‖v‖² is applied
    in turn: O(K·d) per vector, exactly orthogonal for any nonzero v."""
    for k in range(V.shape[0]):
        v = V[k]
        coef = 2.0 * (h @ v) / (v @ v + 1e-8)
        h = h - coef.unsqueeze(-1) * v
    return h


@register("ortho")
class OrthoDSV(DSVController):
    """h' = R_l·(h + δ_l), R_l ∈ SO(d).

    Two parametrisations:
        cayley       — exact identity init, exact SO(d), O(d³) per layer (small d).
        householder  — O(K·d), scalable to large d; K reflection vectors.
    """

    def __init__(self, dim, n_layers=1, parametrization: str = "cayley",
                 n_reflections: int = 8, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.parametrization = parametrization
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)
        if parametrization == "cayley":
            self.raw = nn.Parameter(torch.zeros(n_layers, dim, dim))
        elif parametrization == "householder":
            self.V = nn.Parameter(torch.randn(n_layers, n_reflections, dim) * 0.1)
        else:
            raise ValueError(f"Unknown parametrization '{parametrization}'")

    def rotation(self, layer: int) -> torch.Tensor:
        """Materialise R_l (cayley only; for inspection/tests)."""
        if self.parametrization != "cayley":
            raise RuntimeError("rotation() is only available for the cayley parametrisation")
        return _cayley(self.raw[layer])

    @torch.no_grad()
    def set_rotation(self, R: torch.Tensor, layer: int = 0) -> None:
        """Initialise R_l to a given orthogonal matrix (cayley only).

        Inverts the Cayley map: A = (I - R)(I + R)⁻¹ is skew-symmetric and yields
        exactly R. Useful for closed-form (Procrustes) initialisation before
        optional gradient fine-tuning — gradient descent over SO(d) via Cayley+Adam
        is poorly conditioned in high dimension, so warm-starting matters."""
        if self.parametrization != "cayley":
            raise RuntimeError("set_rotation() is only available for the cayley parametrisation")
        d = R.shape[-1]
        eye = torch.eye(d, device=R.device, dtype=R.dtype)
        A = torch.linalg.solve(eye + R, eye - R)        # (I+R)⁻¹(I-R), skew-symmetric
        self.raw[layer].copy_(A.to(self.raw.dtype))

    def transform(self, h, layer=0):
        shifted = h + self.delta[layer]
        if self.parametrization == "cayley":
            return shifted @ _cayley(self.raw[layer]).T
        return _householder(shifted, self.V[layer])


@register("ogated")
class OrthoGatedDSV(OrthoDSV):
    """h' = R_l·(h + σ(h·a_l)·δ_l). Orthogonal alignment + input-dependent shift."""

    def __init__(self, dim, n_layers=1, **kwargs):
        super().__init__(dim, n_layers, **kwargs)
        self.a = nn.Parameter(torch.zeros(n_layers, dim))
        self.b = nn.Parameter(torch.zeros(n_layers))

    def transform(self, h, layer=0):
        gate = torch.sigmoid((h * self.a[layer]).sum(-1, keepdim=True) + self.b[layer])
        shifted = h + gate * self.delta[layer]
        if self.parametrization == "cayley":
            return shifted @ _cayley(self.raw[layer]).T
        return _householder(shifted, self.V[layer])


@register("scalerot")
class ScaleRotDSV(OrthoDSV):
    """h' = R_l·(σ(g_l)⊙h + δ_l). Scale (channel-gate) then rotate: composes the
    scaling and rotation geometries. Tests whether per-channel rescaling before the
    rotation helps over the pure rotation."""

    def __init__(self, dim, n_layers=1, gate_init: float = 4.0, **kwargs):
        super().__init__(dim, n_layers, **kwargs)
        self.g = nn.Parameter(torch.full((n_layers, dim), float(gate_init)))

    def transform(self, h, layer=0):
        shifted = torch.sigmoid(self.g[layer]) * h + self.delta[layer]
        if self.parametrization == "cayley":
            return shifted @ _cayley(self.raw[layer]).T
        return _householder(shifted, self.V[layer])


@register("rotbasis")
class RotBasisDSV(OrthoDSV):
    """h' = R_l·(h + δ_l) + Σ_k softmax(h·G_l)_k · b_l^k. Rotation plus an
    input-dependent mixture of K learned basis (steering) directions: composes the
    rotation and learned-basis geometries."""

    def __init__(self, dim, n_layers=1, n_basis: int = 4, init_scale: float = 1e-3, **kwargs):
        super().__init__(dim, n_layers, **kwargs)
        self.n_basis = n_basis
        self.basis = nn.Parameter(torch.randn(n_layers, n_basis, dim) * init_scale)
        self.gate = nn.Parameter(torch.zeros(n_layers, dim, n_basis))

    def transform(self, h, layer=0):
        shifted = h + self.delta[layer]
        rot = (shifted @ _cayley(self.raw[layer]).T if self.parametrization == "cayley"
               else _householder(shifted, self.V[layer]))
        w = torch.softmax(h @ self.gate[layer], dim=-1)        # (..., K)
        return rot + w @ self.basis[layer]                     # + steering mixture


# ─────────────────────────────────────────────────────────────────────────────
# LoRA / adapter baselines (what ortho must beat on params-vs-accuracy)
# ─────────────────────────────────────────────────────────────────────────────

@register("lowrank")
class LowRankDSV(DSVController):
    """h' = h + B_l(A_lᵀ h). LoRA applied to activations (rank r)."""

    def __init__(self, dim, n_layers=1, rank: int = 4, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.rank = rank
        self.A = nn.Parameter(torch.randn(n_layers, dim, rank) * init_scale)
        self.B = nn.Parameter(torch.zeros(n_layers, dim, rank))

    def transform(self, h, layer=0):
        return h + (h @ self.A[layer]) @ self.B[layer].T


@register("affine")
class AffineDSV(DSVController):
    """h' = h + W_l h + δ_l. Full (dense) linear adapter."""

    def __init__(self, dim, n_layers=1, init_scale: float = 1e-3):
        super().__init__(dim, n_layers)
        self.W = nn.Parameter(torch.zeros(n_layers, dim, dim))
        self.delta = nn.Parameter(torch.randn(n_layers, dim) * init_scale)

    def transform(self, h, layer=0):
        return h + h @ self.W[layer].T + self.delta[layer]
