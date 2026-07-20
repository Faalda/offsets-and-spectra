"""
Stage-3 task: SLURP-TN Tunisian-Arabic intent classification (text branch).

A frozen transformer encodes the transcription; a trainable head predicts the
intent. Adaptation parameters (a DSV-family controller injected per layer, or
LoRA) are compared at matched budgets. This is the standard PEFT setting, scaled
from Stage-2's single linear layer to the full depth of a transformer.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

# A handful of raw label strings are aliases of canonical intents.
_ALIASES = {
    "sendemail": "email_sendemail", "joke": "general_joke",
    "quirky": "general_quirky", "greet": "general_greet",
    "query": "email_query", "addcontact": "email_addcontact",
    "querycontact": "email_querycontact",
}


def _norm(label: str) -> str:
    return _ALIASES.get(label, label)


@dataclass
class IntentSplit:
    texts: list
    labels: list          # int ids


def load_intent(text_col: str = "tun_transcription", data_id: str = "Elyadata/SLURP-TN"):
    """Returns (train, val, test) IntentSplits and a label2id map."""
    raw = {s: load_dataset(data_id, split=s) for s in ("train", "validation", "test")}
    labels_sorted = sorted({_norm(x) for x in raw["train"]["intent"]})
    label2id = {l: i for i, l in enumerate(labels_sorted)}

    def to_split(ds):
        texts, labels = [], []
        for t, intent in zip(ds[text_col], ds["intent"]):
            intent = _norm(intent)
            if not t or intent not in label2id:
                continue
            texts.append(str(t))
            labels.append(label2id[intent])
        return IntentSplit(texts, labels)

    return to_split(raw["train"]), to_split(raw["validation"]), to_split(raw["test"]), label2id
