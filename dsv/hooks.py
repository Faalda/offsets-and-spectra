"""
Inject a DSV controller into a frozen transformer via forward hooks.

Each selected layer's output hidden state `h` is replaced by `controller(h, idx)`,
where `idx` is the layer's position in the injected list. Because intermediate
layers feed downstream self-attention + FFN (nonlinear), a per-layer transform —
even an orthogonal one — is *not* absorbed by a final linear head: this is what
makes per-layer injection a genuine adaptation rather than a no-op.

    layers = bert_encoder_layers(model)          # the nn.Module list
    ctrl   = build_controller("ortho", dim=768, n_layers=len(layers), ...)
    handles = inject(layers, ctrl)
    ... train / eval ...
    remove(handles)
"""
from __future__ import annotations

import torch.nn as nn


def _make_hook(controller, idx: int):
    def hook(module, inputs, output):
        # BERT-style layers return a tuple (hidden_states, *rest); others a tensor.
        if isinstance(output, tuple):
            return (controller(output[0], idx),) + tuple(output[1:])
        return controller(output, idx)
    return hook


def inject(layers, controller):
    """Register `controller` on each module in `layers`; returns hook handles."""
    return [layer.register_forward_hook(_make_hook(controller, i))
            for i, layer in enumerate(layers)]


def remove(handles):
    for h in handles:
        h.remove()


def encoder_layers(model) -> list:
    """Return the list of transformer encoder layer modules.

    Handles BERT/RoBERTa (`encoder.layer`) and wav2vec2/Whisper (`encoder.layers`)."""
    base = getattr(model, "base_model", model)
    enc = getattr(base, "encoder", None)
    if enc is not None:
        for attr in ("layer", "layers"):
            mods = getattr(enc, attr, None)
            if mods is not None:
                return list(mods)
    raise AttributeError("Could not locate encoder.layer(s) on the model")


# backward-compatible alias
bert_encoder_layers = encoder_layers
