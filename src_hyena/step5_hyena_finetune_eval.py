"""
HyenaDNA  |  Step 5: LoRA Fine-tuning Evaluation
================================================

Hyena-specific LoRA fine-tuning entrypoint for GUE and GUE+ EPI.

This script reuses the GUE/GUE+ data loading, task loop, metrics, result
aggregation and early-stopping protocol from ``src/step5_full_finetune_eval.py``.
Only the per-task training function is replaced: each task gets a fresh
deep-copied HyenaDNA backbone with LoRA adapters and a linear classification
head, while the original backbone weights remain frozen.

Model spec format:

    name:path:mode

where ``mode`` is normally ``encoder`` for HyenaDNA checkpoints. ``causal`` and
``bidir`` are accepted as aliases for consistency with the full fine-tuning
script.

Example:

    uv run python src_hyena/step5_hyena_finetune_eval.py \\
      --models \\
        "H0:LongSafari/hyenadna-small-32k-seqlen-hf:encoder" \\
        "H5:./hyena_h5_crop_contrastive_ep1_s42_r1:encoder" \\
      --gue-plus-only \\
      --gue-plus-dir ./data/GUE_plus \\
      --epi-crop-mode junction \\
      --epi-crop-bp 8192 \\
      --max-length 8192 \\
      --epochs 20 \\
      --batch-size 4 \\
      --lr 2e-4 \\
      --warmup-steps 50 \\
      --monitor mcc \\
      --grad-ckpt \\
      --no-wandb \\
      --output ./eval_results/step5lora_hyena_epi_s42.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
HYENA_SRC = os.path.join(ROOT, "src_hyena")
sys.path.insert(0, SRC)
sys.path.insert(0, HYENA_SRC)

import step5_full_finetune_eval as base  # noqa: E402
from common import extract_hidden_states  # noqa: E402
from hyena_bidirectional import maybe_activate_hyenadna_bidirectional  # noqa: E402
from step5_hyena_full_finetune_eval import (  # noqa: E402
    HyenaModelSpec,
    HyenaSeqDataset,
    build_classification_head,
    load_base_model,
    pool_hyena_hidden,
)

_ORIGINAL_PRINT_RESULTS_TABLE = base.print_results_table


def build_hyena_lora_config(args) -> LoraConfig:
    targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    if not targets:
        raise ValueError("--lora-target-modules must contain at least one module name.")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )


def apply_hyena_lora(model, args):
    peft_model = get_peft_model(model, build_hyena_lora_config(args))
    peft_model.print_trainable_parameters()
    return peft_model


def _peft_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _infer_hidden_dim(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("n_embd", "hidden_size", "d_model", "dim"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    base_model = _peft_base_model(model)
    cfg = getattr(base_model, "config", None)
    for attr in ("n_embd", "hidden_size", "d_model", "dim"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    embeddings = base_model.get_input_embeddings() if hasattr(base_model, "get_input_embeddings") else None
    if embeddings is not None and hasattr(embeddings, "weight"):
        return int(embeddings.weight.shape[1])
    raise RuntimeError("Could not infer HyenaDNA hidden dimension.")


def promote_trainable_parameters_to_fp32(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        dtype_name = str(param.dtype).replace("torch.", "")
        counts[dtype_name] = counts.get(dtype_name, 0) + param.numel()
        if param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.float()
    return counts


class HyenaDNALoRAClassifier(nn.Module):
    """Pool Hyena hidden states from a PEFT-wrapped backbone."""

    def __init__(
        self,
        backbone,
        n_classes: int,
        hidden_dim: int,
        pooling: str = "average",
        head_type: str = "linear",
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = maybe_activate_hyenadna_bidirectional(backbone)
        self.pooling = pooling
        self.classifier = build_classification_head(hidden_dim, n_classes, head_type, head_dropout)

    def forward(self, input_ids, attention_mask):
        model = _peft_base_model(self.backbone)
        core = getattr(model, "hyena", model)
        out = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)
        pooled = pool_hyena_hidden(hidden, attention_mask, self.pooling)
        head_dtype = next(self.classifier.parameters()).dtype
        pooled = pooled.to(dtype=head_dtype)
        return self.classifier(pooled)


def train_one_task(
    base_model,
    tokenizer,
    train_seqs,
    train_labels,
    test_seqs,
    test_labels,
    source: str,
    device: str,
    args,
) -> dict:
    """
    Fine-tune LoRA adapters plus a task head on one GUE/GUE+ task.
    """
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
            train_seqs,
            train_y,
            test_size=0.1,
            random_state=args.seed,
            stratify=stratify,
        )
    else:
        fit_seqs, fit_y = train_seqs, train_y
        val_seqs, val_y = test_seqs, test_y

    train_ds = HyenaSeqDataset(fit_seqs, fit_y, tokenizer, args.max_length)
    val_ds = HyenaSeqDataset(val_seqs, val_y, tokenizer, args.max_length)
    test_ds = HyenaSeqDataset(test_seqs, test_y, tokenizer, args.max_length)
    pad_id = tokenizer.pad_token_id
    collate = lambda b: base._collate_fn(b, pad_id)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
        generator=generator,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
    )

    backbone = copy.deepcopy(base_model)
    backbone = apply_hyena_lora(backbone, args)

    if args.grad_ckpt:
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable({"use_reentrant": False})

    hidden_dim = _infer_hidden_dim(backbone)
    dtype = next(backbone.parameters()).dtype
    classifier = HyenaDNALoRAClassifier(
        backbone,
        n_cls,
        hidden_dim,
        pooling=args.pooling,
        head_type=args.head_type,
        head_dropout=args.head_dropout,
    ).to(device=device, dtype=dtype)
    if args.trainable_fp32:
        before_counts = promote_trainable_parameters_to_fp32(classifier)
        print(
            "    promoted trainable params to fp32 "
            f"(before: {', '.join(f'{k}={v/1e6:.2f}M' for k, v in sorted(before_counts.items()))})"
        )

    trainable = [p for p in classifier.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in classifier.parameters())
    print(
        f"    trainable params={n_trainable/1e6:.2f}M / "
        f"total params={n_total/1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(train_dl) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.warmup_steps >= 0:
        warmup = min(args.warmup_steps, max(1, total_steps - 1))
    else:
        warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    monitor_name = args.monitor
    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = classifier(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            loss = loss / args.grad_accum
            loss.backward()

            if step_idx % args.grad_accum == 0 or step_idx == len(train_dl):
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            del input_ids, attention_mask, labels, logits, loss

        avg_loss = total_loss / len(train_dl)

        classifier.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        with torch.inference_mode():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = classifier(input_ids, attention_mask)
                val_loss = F.cross_entropy(logits, labels)
                total_val_loss += val_loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                del input_ids, attention_mask, labels, logits, val_loss, preds

        avg_val_loss = total_val_loss / len(val_dl)
        acc = float(accuracy_score(all_labels, all_preds))
        f1 = float(f1_score(all_labels, all_preds, average="macro"))
        mcc = float(matthews_corrcoef(all_labels, all_preds))
        metrics = {"accuracy": acc, "f1": f1, "mcc": mcc}
        monitor_val = metrics[monitor_name]
        current_lr = float(scheduler.get_last_lr()[0])

        improved = monitor_val > (best_metric + args.min_delta)
        if improved:
            best_metric = monitor_val
            best_state = {k: v.detach().cpu() for k, v in classifier.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(avg_loss),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(acc),
                "val_f1": float(f1),
                "val_mcc": float(mcc),
                "monitor": monitor_name,
                "monitor_value": float(monitor_val),
                "lr": current_lr,
                "is_best": bool(improved),
            }
        )

        print(
            f"    epoch {epoch+1:02d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"acc={acc*100:.2f}%  F1={f1*100:.2f}%  "
            f"MCC={mcc*100:.2f}%  lr={current_lr:.2e}  "
            f"[monitor={monitor_name}:{monitor_val*100:.2f}%]"
            + ("  *best" if improved else "")
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"    early stop at epoch {epoch+1:02d}  "
                f"(best {monitor_name} at epoch {best_epoch:02d})"
            )
            break

    if best_state is not None:
        classifier.load_state_dict(best_state)

    classifier.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = classifier(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            del input_ids, attention_mask, labels, logits, preds

    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    mcc = float(matthews_corrcoef(all_labels, all_preds))

    del classifier, backbone, optimizer, scheduler, trainable
    del train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, generator
    del all_preds, all_labels, best_state, le, train_y, test_y, fit_y, val_y
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "reported_epoch": int(best_epoch),
        "selection_mode": "best",
        "peak_epoch": int(best_epoch),
        "best_monitor": monitor_name,
        "best_monitor_value": float(best_metric),
        "history": history,
    }


def parse_args():
    p = argparse.ArgumentParser(description="HyenaDNA Step 5: LoRA Fine-tuning Evaluation")

    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode")
    p.add_argument("--task-keys", nargs="+", default=None)

    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--gue-only", action="store_true")
    suite.add_argument("--gue-plus-only", action="store_true")

    p.add_argument("--output", default="./eval_results/step5_hyena_lora_results.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
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
    p.add_argument("--pooling", choices=("average", "last", "weighted"), default="average")
    p.add_argument("--head-type", choices=("linear", "mlp"), default="linear")
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument(
        "--no-trainable-fp32",
        dest="trainable_fp32",
        action="store_false",
        help="Keep LoRA adapters and classifier head in the backbone dtype instead of promoting them to fp32.",
    )
    p.set_defaults(trainable_fp32=True)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--monitor", choices=("f1", "mcc", "accuracy"), default="mcc")
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--lora-target-modules",
        default="in_proj,out_proj,fc1,fc2",
        help="Comma-separated HyenaDNA module-name suffixes for LoRA.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--repeat-index", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--gue-plus-dir", default="")
    p.add_argument("--epi-subdir-names", nargs=6, default=None, metavar=("S0", "S1", "S2", "S3", "S4", "S5"))
    p.add_argument("--epi-crop-bp", type=int, default=8192)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--filter-n", action="store_true")
    args = p.parse_args()

    # The shared main() branches over these flags.
    args.gb_only = False
    args.nt_only = False
    if not args.gue_only and not args.gue_plus_only:
        args.gue_only = True
    return args


def print_results_table(results: dict, benchmarks: list, title: str, metric: str = "accuracy"):
    short_names = [b[2] for b in benchmarks]
    model_names = list(results.keys())
    col_w = max(max(len(n) for n in model_names), 6)
    task_w = 13

    print(f"\n{title}  [metric: {metric}, HyenaDNA LoRA fine-tuning]")
    header = (
        f"{'Model':<{col_w}}  "
        + "  ".join(f"{n:>{task_w}}" for n in short_names)
        + f"  {'Avg':>{task_w}}"
    )
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)

    for model_name, task_map in results.items():
        vals = [base._metric_val(task_map.get(b[0], {}), metric) for b in benchmarks]
        avg = np.nanmean(vals)
        row = f"{model_name:<{col_w}}  "
        row += "  ".join(f"{v*100:>{task_w}.2f}" for v in vals)
        row += f"  {avg*100:>{task_w}.2f}"
        print(row)

    print(sep)
    print(f"({metric} %, LoRA fine-tuning, GB/NT use train->val for selection then test; GUE/GUE+ report dev)")


base.ModelSpec = HyenaModelSpec
base.SeqDataset = HyenaSeqDataset
base.DNAClassifier = HyenaDNALoRAClassifier
base.load_base_model = load_base_model
base.train_one_task = train_one_task
base.parse_args = parse_args
base.print_results_table = print_results_table


if __name__ == "__main__":
    base.main()
