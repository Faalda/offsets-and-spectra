"""
SST-2 sentiment classification (GLUE) — the canonical PEFT 'hello world'.

Frozen encoder + trainable head; adapters compared at matched budgets. The GLUE
test split is unlabeled, so we evaluate on the validation split (standard
practice for reporting on SST-2 dev).
"""
from __future__ import annotations

from datasets import load_dataset

from .intent import IntentSplit


def load_sst2(max_train: int | None = 10000, seed: int = 0):
    """Returns (train, val, test, label2id). `test` mirrors val (GLUE test is unlabeled)."""
    ds = load_dataset("glue", "sst2")
    label2id = {"negative": 0, "positive": 1}

    def to_split(split, cap=None):
        d = ds[split]
        if cap is not None and len(d) > cap:
            d = d.shuffle(seed=seed).select(range(cap))
        return IntentSplit([str(s) for s in d["sentence"]], list(d["label"]))

    train = to_split("train", max_train)
    val = to_split("validation")
    return train, val, val, label2id
