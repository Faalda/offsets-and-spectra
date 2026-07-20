
"""
Stage-5 experiment: multimodal fusion with a SEPARATE DSV controller per modality.

Mirrors the NADI CM-DSV architecture inside the clean DSV package: a frozen speech
encoder (wav2vec2) and a frozen text encoder (BERT), each adapted by its *own*
per-layer controller, with per-modality heads fused at the logit level and a
consistency loss pulling the two projection spaces together.

    frozen wav2vec2 + DSV_speech ─► head_speech ┐
                                                 ├─► α·a+(1-α)·t ─► CE(intent)
    frozen BERT     + DSV_text   ─► head_text   ┘
                       + λ·‖ẑ_speech − ẑ_text‖²  (consistency)

We vary the per-modality controller (none / dsv / ortho / ogated) and compare to
LoRA on both backbones at matched budget. Unlike Stage-4 retrieval, this is the
*supervised* multimodal setting — the regime where the DSV family is competitive
with LoRA (see Stage 3), now extended to two modalities.

Run:
    nadi/.venv/bin/python3 experiments/stage5_multimodal.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsv import build_controller, inject, remove, encoder_layers   # noqa: E402
from tasks.intent import _norm                                     # noqa: E402

SR = 16000
MAX_SAMPLES = SR * 5           # SLURP commands are short; caps the speech graph size
ORTHO_REFLECTIONS = 8
INJECT_LAST = 6                # adapt only the top-N layers per backbone — injecting all
                               # 24 wav2vec2 layers retains the full forward graph (OOM)
PROJ_DIM = 256


class SlurpMM(Dataset):
    def __init__(self, split, label2id, text_col="tun_transcription"):
        from datasets import load_dataset
        ds = load_dataset("Elyadata/SLURP-TN", split=split)
        self.items = []
        for ex in ds:
            intent = _norm(ex["intent"])
            t = ex.get(text_col)
            if not t or intent not in label2id:
                continue
            self.items.append((ex["audio"]["array"], str(t), label2id[intent]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(fe, tok):
    def collate(batch):
        arrays = [torch.as_tensor(a, dtype=torch.float32)[:MAX_SAMPLES].numpy() for a, _, _ in batch]
        texts = [t for _, t, _ in batch]
        labels = torch.tensor([y for _, _, y in batch], dtype=torch.long)
        sp = fe(arrays, sampling_rate=SR, return_tensors="pt", padding=True)
        tx = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        return sp, tx, labels
    return collate


class CrossModalModel(nn.Module):
    def __init__(self, speech, text, sdim, tdim, num_classes, alpha=0.5):
        super().__init__()
        self.speech, self.text, self.alpha = speech, text, alpha
        self.head_s = nn.Linear(sdim, num_classes)
        self.head_t = nn.Linear(tdim, num_classes)
        self.proj_s = nn.Linear(sdim, PROJ_DIM)
        self.proj_t = nn.Linear(tdim, PROJ_DIM)

    @staticmethod
    def _pool(hidden, mask=None):
        if mask is None:
            return hidden.mean(1)
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, sp, tx):
        hs = self.speech(sp["input_values"], attention_mask=sp.get("attention_mask"),
                         return_dict=True).last_hidden_state
        ht = self.text(**tx, return_dict=True).last_hidden_state
        zs = self._pool(hs)                                   # wav2vec2: pool all frames
        zt = self._pool(ht, tx["attention_mask"])
        logits = self.alpha * self.head_s(zs) + (1 - self.alpha) * self.head_t(zt)
        return logits, self.proj_s(zs), self.proj_t(zt)


def build(method, speech_id, text_id, num_classes, device):
    from transformers import AutoModel
    speech = AutoModel.from_pretrained(speech_id).to(device)
    text = AutoModel.from_pretrained(text_id).to(device)
    for m in (speech, text):
        for p in m.parameters():
            p.requires_grad_(False)
    sdim, tdim = speech.config.hidden_size, text.config.hidden_size
    handles, adapter, n_adapter = [], [], 0

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        speech = get_peft_model(speech, LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none"))  # task_type omitted: Wav2Vec2Model has no input embeddings
        text = get_peft_model(text, LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value"],
            lora_dropout=0.0, bias="none"))  # task_type omitted: Wav2Vec2Model has no input embeddings
        adapter = [p for p in list(speech.parameters()) + list(text.parameters()) if p.requires_grad]
        n_adapter = sum(p.numel() for p in adapter)
    elif method != "none":
        kw = {"parametrization": "householder", "n_reflections": ORTHO_REFLECTIONS} \
            if method in ("ortho", "ogated") else {}
        slayers = encoder_layers(speech)[-INJECT_LAST:]
        tlayers = encoder_layers(text)[-INJECT_LAST:]
        cs = build_controller(method, dim=sdim, n_layers=len(slayers), **kw).to(device)
        ct = build_controller(method, dim=tdim, n_layers=len(tlayers), **kw).to(device)
        handles = inject(slayers, cs) + inject(tlayers, ct)
        adapter = list(cs.parameters()) + list(ct.parameters())
        n_adapter = sum(p.numel() for p in adapter)
        build._ctrls = (cs, ct)                               # keep refs alive

    model = CrossModalModel(speech, text, sdim, tdim, num_classes).to(device)
    return model, adapter, handles, n_adapter


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for sp, tx, y in loader:
        sp = {k: v.to(device) for k, v in sp.items()}
        tx = {k: v.to(device) for k, v in tx.items()}
        logits, _, _ = model(sp, tx)
        preds.extend(logits.argmax(-1).cpu().tolist()); gold.extend(y.tolist())
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    return acc, f1_score(gold, preds, average="macro")


def run(method, loaders, args, num_classes, device):
    tl, vl, tel = loaders
    model, adapter, handles, n_adapter = build(method, args.speech_model, args.text_model,
                                               num_classes, device)
    head_params = (list(model.head_s.parameters()) + list(model.head_t.parameters())
                   + list(model.proj_s.parameters()) + list(model.proj_t.parameters()))
    groups = [{"params": head_params, "lr": args.lr_head}]
    if adapter:
        groups.append({"params": adapter, "lr": args.lr_adapter})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    best_vf, best = -1.0, None
    for _ in range(args.epochs):
        model.train()
        for bb in (model.speech, model.text):
            bb.eval()
        for sp, tx, y in tl:
            sp = {k: v.to(device) for k, v in sp.items()}
            tx = {k: v.to(device) for k, v in tx.items()}
            y = y.to(device)
            opt.zero_grad()
            logits, zs, zt = model(sp, tx)
            loss = (F.cross_entropy(logits, y)
                    + args.lambda_consist * F.mse_loss(F.normalize(zs, dim=-1),
                                                        F.normalize(zt, dim=-1)))
            loss.backward()
            opt.step()
        va, vf = evaluate(model, vl, device)
        if vf > best_vf:
            best_vf = vf
            best = (va, vf) + evaluate(model, tel, device)
    remove(handles)
    del model
    torch.cuda.empty_cache()
    return n_adapter, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speech-model", default="jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
    p.add_argument("--text-model", default="CAMeL-Lab/bert-base-arabic-camelbert-da")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-adapter", type=float, default=5e-4)
    p.add_argument("--lambda-consist", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--methods", nargs="*", default=["none", "dsv", "ortho", "ogated", "lora"])
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoFeatureExtractor, AutoTokenizer
    fe = AutoFeatureExtractor.from_pretrained(args.speech_model)
    tok = AutoTokenizer.from_pretrained(args.text_model)

    from tasks.intent import load_intent
    _, _, _, label2id = load_intent()
    num_classes = len(label2id)
    collate = make_collate(fe, tok)
    loaders = tuple(
        DataLoader(SlurpMM(s, label2id), batch_size=args.batch_size,
                   shuffle=(s == "train"), collate_fn=collate)
        for s in ("train", "validation", "test"))

    print(f"\nStage-5 multimodal  | speech {args.speech_model} + text {args.text_model} (both frozen)")
    print(f"SLURP-TN intent | {num_classes} classes | consistency λ={args.lambda_consist}")
    print("-" * 66)
    print(f"{'method':<10}{'adapter params':>16}{'val MF1':>10}{'test acc':>11}{'test MF1':>11}")
    print("-" * 66)
    for method in args.methods:
        try:
            n, (va, vf, ta, tf) = run(method, loaders, args, num_classes, device)
            print(f"{method:<10}{n:>16,}{vf:>10.3f}{ta:>11.3f}{tf:>11.3f}", flush=True)
        except Exception as e:                       # one method failing shouldn't sink the run
            torch.cuda.empty_cache()
            print(f"{method:<10}  FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
    print("-" * 66)


if __name__ == "__main__":
    main()
