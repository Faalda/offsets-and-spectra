"""
Standalone optimization diagnostic for Stage-3 PEFT.

This script keeps the same frozen-backbone setup as stage3_peft.py and compares
DSV-family controllers vs LoRA under matched optimizer-step budgets.

Questions it probes:
1) At very small update budgets (1, 5, 20 steps), how quickly does each method adapt?
2) How sensitive are methods to optimizer choice (AdamW vs SGD variants)?

It is intentionally separate from the main Stage-3 script so paper-critical runs
remain unchanged.

Example:
    nadi/.venv/bin/python3 experiments/stage3_sgd_diagnostic.py \
        --task intent \
        --methods dsv ortho ogated lora \
        --optimizers adamw sgd sgdm \
        --step-budgets 1 5 20 -1 \
        --seeds 0 1 2 \
        --out runs/stage3_sgd_diag_intent.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import bert_encoder_layers, build_controller, inject, remove  # noqa: E402
from tasks.intent import load_intent  # noqa: E402

ORTHO_REFLECTIONS = 8


class FrozenClassifier(nn.Module):
    """Frozen encoder + mean-pool + trainable linear head."""

    def __init__(self, backbone, dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split, tok, batch_size, shuffle, seed):
    enc = tok(split.texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    ds = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(split.labels, dtype=torch.long),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for ids, mask, y in loader:
        logits = model(ids.to(device), mask.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(y.tolist())
    acc = sum(p == g for p, g in zip(preds, gold)) / max(1, len(gold))
    return acc, f1_score(gold, preds, average="macro")


def build_method(name, model_id, dim, num_classes, n_layers, device):
    """Return classifier, adapter params, handles, adapter param count."""
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_id).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    handles, adapter_params, n_adapter = [], [], 0

    if name == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, cfg)
        adapter_params = [p for p in backbone.parameters() if p.requires_grad]
        n_adapter = sum(p.numel() for p in adapter_params)
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
    elif name == "none":
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
    else:
        kw = (
            {"parametrization": "householder", "n_reflections": ORTHO_REFLECTIONS}
            if name in ("ortho", "ogated")
            else ({"n_experts": 16} if name == "mixture" else {})
        )
        ctrl = build_controller(name, dim=dim, n_layers=n_layers, **kw).to(device)
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
        handles = inject(bert_encoder_layers(backbone), ctrl)
        adapter_params = list(ctrl.parameters())
        n_adapter = ctrl.num_params()
        clf._ctrl = ctrl

    return clf, adapter_params, handles, n_adapter


def build_optimizer(opt_name, head_params, adapter_params, lr_head, lr_adapter, wd):
    groups = [{"params": head_params, "lr": lr_head}]
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr_adapter})

    if opt_name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(groups, momentum=0.0, weight_decay=wd)
    if opt_name == "sgdm":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=False, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{opt_name}'")


def train_with_budget(clf, opt, tl, device, epochs, step_budget):
    """Train for either full epochs (step_budget < 0) or fixed optimizer steps."""
    steps = 0
    total_target = math.inf if step_budget < 0 else step_budget

    for _ in range(epochs):
        clf.train()
        clf.backbone.eval()
        for ids, mask, y in tl:
            if steps >= total_target:
                return steps
            opt.zero_grad()
            logits = clf(ids.to(device), mask.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            opt.step()
            steps += 1
        if step_budget < 0:
            continue
    return steps


def run_once(method, optimizer_name, step_budget, splits, tok, args, device, seed):
    set_all_seeds(seed)

    train, val, test = splits
    tl = make_loader(train, tok, args.batch_size, True, seed)
    vl = make_loader(val, tok, args.batch_size, False, seed)
    tel = make_loader(test, tok, args.batch_size, False, seed)

    clf, adapter_params, handles, n_adapter = build_method(
        method,
        args.model,
        args.dim,
        args.num_classes,
        args.n_layers,
        device,
    )

    try:
        opt = build_optimizer(
            optimizer_name,
            list(clf.head.parameters()),
            adapter_params,
            args.lr_head,
            args.lr_adapter,
            args.weight_decay,
        )
        n_steps = train_with_budget(clf, opt, tl, device, args.epochs, step_budget)
        va, vf = evaluate(clf, vl, device)
        ta, tf = evaluate(clf, tel, device)
    finally:
        remove(handles)
        del clf
        torch.cuda.empty_cache()

    return {
        "method": method,
        "optimizer": optimizer_name,
        "step_budget": step_budget,
        "steps_run": n_steps,
        "adapter_params": n_adapter,
        "val_acc": va,
        "val_mf1": vf,
        "test_acc": ta,
        "test_mf1": tf,
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["intent", "sst2"], default="intent")
    p.add_argument(
        "--model",
        default=None,
        help="Backbone model. Defaults per task (CAMeL-BERT for intent, roberta-base for sst2).",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-adapter", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--methods", nargs="*", default=["dsv", "ortho", "ogated", "lora"])
    p.add_argument("--optimizers", nargs="*", default=["adamw", "sgd", "sgdm"])
    p.add_argument(
        "--step-budgets",
        nargs="*",
        type=int,
        default=[1, 5, 20, -1],
        help="Optimizer step budgets; use -1 for full training over --epochs.",
    )
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--out", default=None, help="Optional path for JSON output.")
    args = p.parse_args()

    if args.model is None:
        args.model = (
            "CAMeL-Lab/bert-base-arabic-camelbert-da"
            if args.task == "intent"
            else "roberta-base"
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.task == "intent":
        train, val, test, label2id = load_intent()
    else:
        from tasks.sst2 import load_sst2

        train, val, test, label2id = load_sst2(max_train=args.max_train)

    cfg = AutoConfig.from_pretrained(args.model)
    args.n_layers = cfg.num_hidden_layers
    args.dim = cfg.hidden_size
    args.num_classes = len(label2id)

    print(
        f"\nStage-3 SGD diagnostic [{args.task}] | {args.model} "
        f"(frozen, {args.n_layers}L, d={args.dim})"
    )
    print(f"methods={args.methods}")
    print(f"optimizers={args.optimizers} | budgets={args.step_budgets} | seeds={args.seeds}")
    print("-" * 96)
    print(
        f"{'method':<10}{'opt':<8}{'budget':>8}{'seed':>6}{'steps':>8}"
        f"{'params':>12}{'val MF1':>10}{'test acc':>11}{'test MF1':>11}"
    )
    print("-" * 96)

    rows = []
    splits = (train, val, test)
    for method in args.methods:
        for opt_name in args.optimizers:
            for budget in args.step_budgets:
                for seed in args.seeds:
                    row = run_once(method, opt_name, budget, splits, tok, args, device, seed)
                    rows.append(row)
                    print(
                        f"{row['method']:<10}{row['optimizer']:<8}{row['step_budget']:>8}"
                        f"{row['seed']:>6}{row['steps_run']:>8}{row['adapter_params']:>12,}"
                        f"{row['val_mf1']:>10.3f}{row['test_acc']:>11.3f}{row['test_mf1']:>11.3f}"
                    )

    print("-" * 96)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "model": args.model,
            "n_layers": args.n_layers,
            "hidden_size": args.dim,
            "num_classes": args.num_classes,
            "methods": args.methods,
            "optimizers": args.optimizers,
            "step_budgets": args.step_budgets,
            "seeds": args.seeds,
            "rows": rows,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
