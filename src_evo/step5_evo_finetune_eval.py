"""
Evo Step 5: task fine-tuning with LoRA adapters.

This mirrors the HyenaDNA LoRA Step 5 wrapper: benchmark loading and result
tables come from ``src/step5_full_finetune_eval.py``; Evo-specific code handles
loading, tokenisation, hidden-state pooling, and PEFT LoRA attachment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(__file__))

import step5_full_finetune_eval as base  # noqa: E402
from common import (  # noqa: E402
    count_parameters_m,
    evo_hidden_states,
    infer_hidden_dim,
    infer_lora_target_modules,
    load_evo_causal_lm,
    load_evo_tokenizer,
    mean_pool_embeddings,
)


@dataclass
class EvoModelSpec:
    name: str
    path: str
    mode: str

    @staticmethod
    def parse(spec: str) -> "EvoModelSpec":
        parts = spec.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError("Expected model spec name:path:mode, e.g. E5:./evo_e5_crop_contrastive_lora:evo")
        name, path, mode = parts
        if mode not in ("evo", "causal", "bidir", "encoder"):
            raise ValueError(f"Unsupported Evo mode {mode!r}.")
        return EvoModelSpec(name, path, mode)


class EvoSeqDataset(base.Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length: int):
        enc = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        self.input_ids = enc["input_ids"]
        if "attention_mask" in enc:
            self.attention_mask = enc["attention_mask"]
        else:
            pad_id = tokenizer.pad_token_id
            self.attention_mask = [[1 if pad_id is None or tok != pad_id else 0 for tok in x] for x in self.input_ids]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_lora_config(model, args):
    targets = infer_lora_target_modules(model, args.lora_target_modules)
    print(f"    LoRA targets={targets}")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )


class EvoLoRAClassifier(nn.Module):
    def __init__(self, backbone, n_classes: int, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden = evo_hidden_states(self.backbone, input_ids, attention_mask)
        pooled = mean_pool_embeddings(hidden, attention_mask)
        return self.classifier(pooled)


def load_base_model(spec: EvoModelSpec, device: str, dtype):
    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, evo)")
    tokenizer = load_evo_tokenizer(path)
    model, load_path = load_evo_causal_lm(path, device="cpu", dtype=dtype)
    hidden_dim = infer_hidden_dim(model)
    model.config.n_embd = hidden_dim
    model.eval()
    print(f"    Load path  : {load_path}")
    print(f"    Parameters : {count_parameters_m(model):.1f}M  |  hidden: {hidden_dim}  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


def train_one_task(base_model, tokenizer, train_seqs, train_labels, test_seqs, test_labels, source, device, args):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    train_y = le.fit_transform(train_labels)
    test_y = le.transform(test_labels)
    n_cls = len(le.classes_)

    if source in ("gb", "nt"):
        stratify = train_y if len(set(train_y)) > 1 else None
        fit_seqs, val_seqs, fit_y, val_y = train_test_split(
            train_seqs, train_y, test_size=0.1, random_state=args.seed, stratify=stratify
        )
    else:
        fit_seqs, fit_y = train_seqs, train_y
        val_seqs, val_y = test_seqs, test_y

    train_ds = EvoSeqDataset(fit_seqs, fit_y, tokenizer, args.max_length)
    val_ds = EvoSeqDataset(val_seqs, val_y, tokenizer, args.max_length)
    test_ds = EvoSeqDataset(test_seqs, test_y, tokenizer, args.max_length)
    collate = lambda b: base._collate_fn(b, tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    backbone = copy.deepcopy(base_model)
    backbone = get_peft_model(backbone, build_lora_config(backbone, args))
    backbone.print_trainable_parameters()
    if args.grad_ckpt and hasattr(backbone, "gradient_checkpointing_enable"):
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable({"use_reentrant": False})

    hidden_dim = infer_hidden_dim(backbone.base_model.model if hasattr(backbone, "base_model") else backbone)
    dtype = next(backbone.parameters()).dtype
    classifier = EvoLoRAClassifier(backbone, n_cls, hidden_dim).to(device=device, dtype=dtype)
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(train_dl) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps if args.warmup_steps >= 0 else int(total_steps * args.warmup_ratio), max(1, total_steps - 1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = F.cross_entropy(classifier(input_ids, attention_mask), labels)
            total_loss += loss.item()
            (loss / args.grad_accum).backward()
            if step_idx % args.grad_accum == 0 or step_idx == len(train_dl):
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        classifier.eval()
        preds, labels_out = [], []
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = classifier(input_ids, attention_mask)
                val_loss += F.cross_entropy(logits, labels).item()
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                labels_out.extend(labels.cpu().tolist())
        metrics = {
            "accuracy": float(accuracy_score(labels_out, preds)),
            "f1": float(f1_score(labels_out, preds, average="macro")),
            "mcc": float(matthews_corrcoef(labels_out, preds)),
        }
        monitor_val = metrics[args.monitor]
        improved = monitor_val > best_metric + args.min_delta
        if improved:
            best_metric = monitor_val
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_dl), "val_loss": val_loss / len(val_dl), **{f"val_{k}": v for k, v in metrics.items()}, "is_best": improved})
        print(f"    epoch {epoch+1:02d}/{args.epochs} loss={total_loss/len(train_dl):.4f} val_{args.monitor}={monitor_val*100:.2f}%" + (" *best" if improved else ""))
        if args.patience > 0 and stale >= args.patience:
            break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    classifier.eval()
    preds, labels_out = [], []
    with torch.inference_mode():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = classifier(input_ids, attention_mask)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_out.extend(labels.cpu().tolist())
    result = {
        "accuracy": float(accuracy_score(labels_out, preds)),
        "f1": float(f1_score(labels_out, preds, average="macro")),
        "mcc": float(matthews_corrcoef(labels_out, preds)),
        "reported_epoch": int(best_epoch),
        "selection_mode": "best",
        "peak_epoch": int(best_epoch),
        "best_monitor": args.monitor,
        "best_monitor_value": float(best_metric),
        "history": history,
    }
    del classifier, backbone, optimizer, scheduler, trainable
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evo Step 5: LoRA fine-tuning evaluation")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode")
    p.add_argument("--task-keys", nargs="+", default=None)
    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--gue-only", action="store_true")
    suite.add_argument("--gue-plus-only", action="store_true")
    p.add_argument("--output", default="./eval_results/step5_evo_lora_results.json")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--gb-root", default=os.environ.get("GENOMIC_BENCHMARKS_DIR", os.path.expanduser("~/.genomic_benchmarks")))
    p.add_argument("--benchmark-cache-dir", default="./cache/benchmarks")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--monitor", choices=("f1", "mcc", "accuracy"), default="mcc")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--repeat-index", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--gue-plus-dir", default="")
    p.add_argument("--epi-subdir-names", nargs=6, default=None)
    p.add_argument("--epi-crop-bp", type=int, default=8192)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--filter-n", action="store_true")
    args = p.parse_args()
    args.gb_only = False
    args.nt_only = False
    if not args.gue_only and not args.gue_plus_only:
        args.gue_only = True
    return args


base.ModelSpec = EvoModelSpec
base.SeqDataset = EvoSeqDataset
base.DNAClassifier = EvoLoRAClassifier
base.load_base_model = load_base_model
base.train_one_task = train_one_task
base.parse_args = parse_args


if __name__ == "__main__":
    base.main()
