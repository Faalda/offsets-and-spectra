"""
Stage-3 experiment: per-layer adaptation of a FROZEN transformer, DSV family vs LoRA.

A frozen BERT encodes SLURP-TN transcriptions; a trainable linear head predicts
intent. We compare parameter-efficient adapters at matched budgets:

    none      head only (lower bound — how good are the frozen features?)
    dsv       per-layer additive shift          (the original DSV method)
    ortho     per-layer orthogonal transform    (Householder, scalable)
    ogated    ortho + input-dependent gate
    affine    per-layer dense linear adapter
    lora      LoRA on attention q/v             (peft)

This is the gradient-trained regime: unlike Stage-2's single linear layer, the
stacked nonlinear backbone has no closed form, so controllers are trained by
backprop (with the frozen backbone in the loop).

Run:
    nadi/.venv/bin/python3 experiments/stage3_peft.py
    nadi/.venv/bin/python3 experiments/stage3_peft.py --methods none dsv ortho lora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import build_controller, inject, remove, bert_encoder_layers   # noqa: E402
from tasks.intent import load_intent                                    # noqa: E402

ORTHO_REFLECTIONS = 8           # Householder K per layer (param-comparable to LoRA)


class FrozenClassifier(nn.Module):
    """Frozen encoder + mean-pool + trainable linear head."""

    def __init__(self, backbone, dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=True).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)


def make_loader(split, tok, batch_size, shuffle):
    enc = tok(split.texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(split.labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for ids, mask, y in loader:
        logits = model(ids.to(device), mask.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(y.tolist())
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    return acc, f1_score(gold, preds, average="macro")


def build_method(name, model_id, dim, num_classes, n_layers, device):
    """Returns (classifier, adapter_params_list, hook_handles, adapter_param_count)."""
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(model_id).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    handles, adapter_params, n_adapter = [], [], 0

    if name == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                         lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
        backbone = get_peft_model(backbone, cfg)
        adapter_params = [p for p in backbone.parameters() if p.requires_grad]
        n_adapter = sum(p.numel() for p in adapter_params)
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
    elif name == "none":
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
    else:
        kw = {"parametrization": "householder", "n_reflections": ORTHO_REFLECTIONS} \
            if name in ("ortho", "ogated") else ({"n_experts": 16} if name == "mixture" else {})
        ctrl = build_controller(name, dim=dim, n_layers=n_layers, **kw).to(device)
        clf = FrozenClassifier(backbone, dim, num_classes).to(device)
        handles = inject(bert_encoder_layers(backbone), ctrl)
        adapter_params = list(ctrl.parameters())
        n_adapter = ctrl.num_params()
        clf._ctrl = ctrl                       # keep a reference so it isn't GC'd
    return clf, adapter_params, handles, n_adapter


def train_method(name, splits, tok, args, device):
    train, val, test = splits
    dim = args.dim
    num_classes = args.num_classes
    n_layers = args.n_layers
    clf, adapter_params, handles, n_adapter = build_method(
        name, args.model, dim, num_classes, n_layers, device)

    params = [{"params": clf.head.parameters(), "lr": args.lr_head}]
    if adapter_params:
        params.append({"params": adapter_params, "lr": args.lr_adapter})
    opt = torch.optim.AdamW(params, weight_decay=1e-2)

    tl = make_loader(train, tok, args.batch_size, True)
    vl = make_loader(val, tok, args.batch_size, False)
    tel = make_loader(test, tok, args.batch_size, False)

    best_val_f1, best = -1.0, None
    for epoch in range(args.epochs):
        clf.train(); clf.backbone.eval()
        for ids, mask, y in tl:
            opt.zero_grad()
            loss = F.cross_entropy(clf(ids.to(device), mask.to(device)), y.to(device))
            loss.backward()
            opt.step()
        va, vf = evaluate(clf, vl, device)
        if vf > best_val_f1:
            best_val_f1 = vf
            best = (va, vf) + evaluate(clf, tel, device)
    remove(handles)
    del clf
    torch.cuda.empty_cache()
    return n_adapter, best          # (adapter_params, (val_acc, val_f1, test_acc, test_f1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["intent", "sst2"], default="intent")
    p.add_argument("--model", default=None,
                   help="backbone; defaults per task (CAMeL-BERT for intent, roberta-base for sst2)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-adapter", type=float, default=5e-4)
    p.add_argument("--max-train", type=int, default=10000, help="cap train size (sst2)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional path to save results JSON")
    p.add_argument("--methods", nargs="*",
                   default=["none", "dsv", "ortho", "ogated", "affine", "lora"])
    args = p.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model is None:
        args.model = ("CAMeL-Lab/bert-base-arabic-camelbert-da" if args.task == "intent"
                      else "roberta-base")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.task == "intent":
        train, val, test, label2id = load_intent()
    else:
        from tasks.sst2 import load_sst2
        train, val, test, label2id = load_sst2(max_train=args.max_train)
    args.num_classes = len(label2id)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    args.n_layers = cfg.num_hidden_layers
    args.dim = cfg.hidden_size

    print(f"\nStage-3 PEFT [{args.task}]  | backbone {args.model} "
          f"(frozen, {args.n_layers}L, d={args.dim})")
    print(f"{len(train.texts)}/{len(val.texts)}/{len(test.texts)} "
          f"train/val/test | {args.num_classes} classes")
    print("-" * 66)
    print(f"{'method':<10}{'adapter params':>16}{'val MF1':>10}{'test acc':>11}{'test MF1':>11}")
    print("-" * 66)

    rows = []
    for name in args.methods:
        n_adapter, (va, vf, ta, tf) = train_method(name, (train, val, test), tok, args, device)
        rows.append((name, n_adapter, vf, ta, tf))
    for name, n, vf, ta, tf in rows:
        print(f"{name:<10}{n:>16,}{vf:>10.3f}{ta:>11.3f}{tf:>11.3f}")
    print("-" * 66)

    if args.out:
        import json
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"task": args.task, "model": args.model, "seed": args.seed,
                   "methods": {name: {"params": n, "val_mf1": vf, "test_acc": ta,
                                      "test_mf1": tf} for name, n, vf, ta, tf in rows}},
                  open(args.out, "w"), indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
