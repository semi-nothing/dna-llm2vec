"""
DNA-LLM2Vec  |  Step 5: Full Fine-tuning Evaluation
===================================================
Evaluates DNA model quality via full backbone fine-tuning + classification head,
on the same three benchmark suites as step4 (linear probe):

  GB  — Genomics Benchmarks (8 tasks). Conventional display metric: accuracy.
  NT  — Nucleotide Transformer downstream tasks (18 tasks). Conventional display
        metric: accuracy, while F1 and MCC are also computed.
  GUE — Genome Understanding Evaluation (18 tasks). Metric: F1 (macro) + MCC.
        Matches DNABERT-2 paper convention, enabling direct comparison.

Unlike step4 (frozen model + logistic regression), here we:
  1. Fine-tune all backbone parameters.
  2. Add a trainable mean-pool → Linear classification head.
  3. Optimize backbone + head end-to-end with cross-entropy.
  4. Report test metrics (GB/NT: test split; GUE/GUE+ EPI: dev split).

For each model × task, a fresh full-fine-tuned copy of the base model is trained.
Base model weights are loaded once per model spec and deep-copied per task.

Model spec format  →  name:path:mode
  name  — label in results table (e.g. M0, M4)
  path  — local directory or HuggingFace model ID
  mode  — "causal" | "bidir"

Usage
-----
  # GUE+ EPI full fine-tuning (closest to DNABERT-2-style adaptation):
  uv run python src/step5_full_finetune_eval.py \\
      --models "M0:dnagpt/human_gpt2-v1:causal" \\
      --gue-plus-only --gue-plus-dir ./data/GUE_plus \\
      --epi-crop-bp 4096 --epi-crop-mode junction \\
      --epochs 5 --batch-size 8 --lr 3e-5 --grad-ckpt
"""

import argparse
import copy
import gc
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(__file__))
from step1_bidirectional import patch_to_bidirectional


# ── Benchmark registries (same as step4) ─────────────────────────────────────

GB_BENCHMARKS = [
    ("demo_coding_vs_intergenomic_seqs", 2, "Coding-vs-Intergenic", "gb"),
    ("demo_human_or_worm",               2, "Human-vs-Worm",        "gb"),
    ("human_enhancers_cohn",          2, "Enh-Cohn",      "gb"),
    ("human_enhancers_ensembl",        2, "Enh-Ensembl",   "gb"),
    ("human_ensembl_regulatory",       3, "Regulatory",    "gb"),
    ("human_nontata_promoters",        2, "Non-TATA",      "gb"),
    ("human_ocr_ensembl",              2, "OCR",           "gb"),
    ("dummy_mouse_enhancers_ensembl",  2, "Mouse-Enh",    "gb"),
]

NT_BENCHMARKS = [
    ("promoter_all",          2, "Promoter-All",        "nt"),
    ("promoter_tata",         2, "Promoter-TATA",       "nt"),
    ("promoter_no_tata",      2, "Promoter-noTATA",     "nt"),
    ("enhancers",             2, "Enhancers",           "nt"),
    ("enhancers_types",       3, "Enhancer-Types",      "nt"),
    ("splice_sites_all",      3, "Splice-All",          "nt"),
    ("splice_sites_acceptor", 2, "Splice-Acceptor",     "nt"),
    ("splice_sites_donor",    2, "Splice-Donor",        "nt"),
    ("H3",                    2, "H3",                  "nt"),
    ("H4",                    2, "H4",                  "nt"),
    ("H3K9ac",                2, "H3K9ac",              "nt"),
    ("H3K14ac",               2, "H3K14ac",             "nt"),
    ("H4ac",                  2, "H4ac",                "nt"),
    ("H3K4me1",               2, "H3K4me1",             "nt"),
    ("H3K4me2",               2, "H3K4me2",             "nt"),
    ("H3K4me3",               2, "H3K4me3",             "nt"),
    ("H3K36me3",              2, "H3K36me3",            "nt"),
    ("H3K79me3",              2, "H3K79me3",            "nt"),
]

GUE_BENCHMARKS = [
    ("prom_core_all",        2, "Prom-Core-All",    "gue"),
    ("prom_core_tata",       2, "Prom-Core-TATA",   "gue"),
    ("prom_core_notata",     2, "Prom-Core-noTATA", "gue"),
    ("prom_300_all",         2, "Prom-300-All",     "gue"),
    ("prom_300_tata",        2, "Prom-300-TATA",    "gue"),
    ("prom_300_notata",      2, "Prom-300-noTATA",  "gue"),
    ("human_tf_0",           2, "TF-Human-0",       "gue"),
    ("human_tf_1",           2, "TF-Human-1",       "gue"),
    ("human_tf_2",           2, "TF-Human-2",       "gue"),
    ("human_tf_3",           2, "TF-Human-3",       "gue"),
    ("human_tf_4",           2, "TF-Human-4",       "gue"),
    ("mouse_0",              2, "TF-Mouse-0",       "gue"),
    ("mouse_1",              2, "TF-Mouse-1",       "gue"),
    ("mouse_2",              2, "TF-Mouse-2",       "gue"),
    ("mouse_3",              2, "TF-Mouse-3",       "gue"),
    ("mouse_4",              2, "TF-Mouse-4",       "gue"),
    ("splice_reconstructed", 3, "Splice",           "gue"),
    ("virus_covid",          2, "Virus-COVID",      "gue"),
]

ALL_BENCHMARKS  = GB_BENCHMARKS + NT_BENCHMARKS + GUE_BENCHMARKS
NT_HF_DATASET   = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
GUE_HF_DATASET  = "leannmlindsey/GUE"

NT_TASK_ALIASES = {
    # The NT HF dataset has used plural names for these tasks in some
    # revisions/caches; keep our public result keys stable.
    "splice_sites_acceptor": ("splice_sites_acceptors",),
    "splice_sites_donor": ("splice_sites_donors",),
}

_nt_cache:  dict = {}
_gue_cache: dict = {}
_gb_root = os.environ.get("GENOMIC_BENCHMARKS_DIR", os.path.expanduser("~/.genomic_benchmarks"))
_benchmark_cache_dir = "./cache/benchmarks"

# GUE+ EPI — Enhancer-Promoter Interaction (DNABERT-2 extended benchmark)
# 6 datasets, 5000 bp sequences, binary classification.
# Requires manual download from https://github.com/MAGICS-LAB/DNABERT_2
GUE_PLUS_EPI_BENCHMARKS = [
    ("epi_0", 2, "EPI-GM12878", "gue+"),
    ("epi_1", 2, "EPI-HeLa-S3", "gue+"),
    ("epi_2", 2, "EPI-HUVEC",   "gue+"),
    ("epi_3", 2, "EPI-IMR90",   "gue+"),
    ("epi_4", 2, "EPI-K562",    "gue+"),
    ("epi_5", 2, "EPI-NHEK",    "gue+"),
]
_EPI_SUBDIR_DEFAULT = {
    "epi_0": "GM12878", "epi_1": "HeLa-S3", "epi_2": "HUVEC",
    "epi_3": "IMR90",   "epi_4": "K562",    "epi_5": "NHEK",
}
_epi_subdir_map: dict = {}

def _gb_pickle_path(task_key: str) -> str:
    return os.path.join(_benchmark_cache_dir, f"gb_{task_key}.pkl")

def _gb_task_present(task_key: str) -> bool:
    task_dir = os.path.join(_gb_root, task_key)
    return os.path.isdir(task_dir)


# ── Model spec ────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    path: str
    mode: str   # "causal" | "bidir"

    @staticmethod
    def parse(spec: str) -> "ModelSpec":
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model spec '{spec}'. "
                "Expected: name:path:mode  (e.g. M0:dnagpt/human_gpt2-v1:causal)"
            )
        name, path, mode = parts
        if mode not in ("causal", "bidir"):
            raise ValueError(f"mode must be 'causal' or 'bidir', got '{mode}'")
        return ModelSpec(name=name, path=path, mode=mode)


# ── Data loading (same logic as step4) ───────────────────────────────────────

def _load_gb(task_key: str):
    cache_path = _gb_pickle_path(task_key)
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
    if not _gb_task_present(task_key):
        download_dataset(task_key, version=0)
    def _unpack(split):
        ds = GenomicClfDataset(task_key, split=split)
        seqs, labels = zip(*[(s, l) for s, l in ds])
        return list(seqs), list(labels)
    tr_s, tr_l = _unpack("train")
    te_s, te_l = _unpack("test")
    payload = (tr_s, tr_l, te_s, te_l)
    os.makedirs(_benchmark_cache_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def _load_nt(task_key: str):
    if "ds" not in _nt_cache:
        from datasets import load_dataset
        _nt_cache["ds"] = load_dataset(NT_HF_DATASET)
    ds = _nt_cache["ds"]
    def _filter(split, candidate):
        rows = ds[split].filter(lambda x: x["task"] == candidate)
        return [r["sequence"] for r in rows], [r["label"] for r in rows]
    tr_s, tr_l = _filter("train", task_key)
    te_s, te_l = _filter("test", task_key)
    if (not tr_s or not te_s) and task_key in NT_TASK_ALIASES:
        for alias in NT_TASK_ALIASES[task_key]:
            alias_tr_s, alias_tr_l = _filter("train", alias)
            alias_te_s, alias_te_l = _filter("test", alias)
            if alias_tr_s and alias_te_s:
                print(f"       [nt-alias] {task_key} -> {alias}")
                return alias_tr_s, alias_tr_l, alias_te_s, alias_te_l
    return tr_s, tr_l, te_s, te_l


def _validate_benchmark_splits(task_key: str, source: str, tr_s, tr_l, te_s, te_l) -> None:
    split_label = "dev" if source in ("gue", "gue+") else "test"
    if len(tr_s) != len(tr_l):
        raise ValueError(
            f"{task_key}: train sequence/label count mismatch ({len(tr_s)} vs {len(tr_l)})"
        )
    if len(te_s) != len(te_l):
        raise ValueError(
            f"{task_key}: {split_label} sequence/label count mismatch ({len(te_s)} vs {len(te_l)})"
        )
    if not tr_s or not te_s:
        hint = ""
        if source == "nt":
            hint = f" Queried HF task='{task_key}'. Check the dataset cache/schema if this should be nonempty."
        raise ValueError(
            f"{task_key}: loaded empty benchmark split "
            f"(train={len(tr_s)}, {split_label}={len(te_s)}).{hint}"
        )


def _load_gue(task_key: str):
    from datasets import load_dataset
    if task_key not in _gue_cache:
        _gue_cache[task_key] = load_dataset(GUE_HF_DATASET, name=task_key)
    ds = _gue_cache[task_key]
    return (list(ds["train"]["sequence"]), list(ds["train"]["label"]),
            list(ds["dev"]["sequence"]),   list(ds["dev"]["label"]))


def _crop_sequence(seq: str, max_bp: int, mode: str = "center", anchor: int | None = None) -> str:
    """Crop a DNA sequence to at most max_bp base pairs."""
    if len(seq) <= max_bp:
        return seq
    if mode == "junction":
        if anchor is None:
            anchor = len(seq) // 2
        start = anchor - (max_bp // 2)
        start = max(0, min(start, len(seq) - max_bp))
    elif mode == "center":
        start = (len(seq) - max_bp) // 2
    else:
        raise ValueError(f"Unknown crop mode '{mode}'. Expected: center | junction")
    return seq[start : start + max_bp]


def _is_acgt(seq: str) -> bool:
    return all(base in "ACGT" for base in seq)


def _load_gue_plus(
    task_key: str,
    gue_plus_dir: str,
    crop_bp: int = 4096,
    crop_mode: str = "center",
    filter_non_acgt: bool = False,
):
    """Load one GUE+ EPI task from locally downloaded DNABERT-2 files."""
    import csv as _csv
    subdir = _epi_subdir_map.get(task_key, _EPI_SUBDIR_DEFAULT.get(task_key, task_key))
    task_dir = os.path.join(gue_plus_dir, "EPI", subdir)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(
            f"GUE+ EPI directory not found: {task_dir}\n"
            f"  Download from https://github.com/MAGICS-LAB/DNABERT_2\n"
            f"  and set --gue-plus-dir to the parent of the EPI/ folder."
        )
    def _read(split_name):
        path = os.path.join(task_dir, f"{split_name}.csv")
        seqs, labels = [], []
        dropped = 0
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                seq = row.get("sequence") or row.get("seq") or ""
                anchor = None
                if not seq and "enhancer" in row and "promoter" in row:
                    enhancer = row["enhancer"].strip().upper()
                    promoter = row["promoter"].strip().upper()
                    seq = enhancer + promoter
                    anchor = len(enhancer)
                seq = _crop_sequence(seq.strip().upper(), crop_bp, crop_mode, anchor)
                if filter_non_acgt and not _is_acgt(seq):
                    dropped += 1
                    continue
                seqs.append(seq)
                labels.append(int(row["label"]))
        if dropped:
            print(f"       [filter-n] dropped {dropped} {split_name} rows with non-ACGT bases")
        return seqs, labels
    tr_s, tr_l = _read("train")
    te_s, te_l = _read("dev")
    return tr_s, tr_l, te_s, te_l


def load_benchmark_data(task_key: str, source: str,
                        gue_plus_dir: str = "", crop_bp: int = 4096,
                        crop_mode: str = "center",
                        filter_non_acgt: bool = False):
    if source == "gb":
        payload = _load_gb(task_key)
    elif source == "nt":
        payload = _load_nt(task_key)
    elif source == "gue":
        payload = _load_gue(task_key)
    elif source == "gue+":
        payload = _load_gue_plus(
            task_key,
            gue_plus_dir,
            crop_bp,
            crop_mode,
            filter_non_acgt,
        )
    else:
        raise ValueError(f"Unknown source '{source}'")
    _validate_benchmark_splits(task_key, source, *payload)
    return payload


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """Tokenise a list of sequences — no padding here; pad per batch via collate_fn."""

    def __init__(self, sequences, labels, tokenizer, max_length: int):
        enc = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,          # dynamic padding per batch, not global
            return_tensors=None,    # list of lists, not tensors
        )
        self.input_ids      = enc["input_ids"]        # list[list[int]]
        self.attention_mask = enc["attention_mask"]   # list[list[int]]
        self.labels         = labels                  # list[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


def _collate_fn(batch, pad_token_id: int):
    """Pad each batch to the longest sequence in that batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"])
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


# ── Classification model ──────────────────────────────────────────────────────

class DNAClassifier(nn.Module):
    """Pool backbone hidden states, then apply a task classification head."""

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
        self.backbone = backbone
        self.pooling = pooling
        if head_type == "linear":
            self.classifier = nn.Linear(hidden_dim, n_classes)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError(f"Unsupported head_type={head_type!r}")

    def _pool_hidden(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "average":
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pooling == "weighted":
            positions = torch.arange(
                1, hidden.size(1) + 1, device=hidden.device, dtype=hidden.dtype
            ).view(1, -1, 1)
            weights = mask * positions
            return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        if self.pooling == "last":
            lengths = attention_mask.sum(dim=1).clamp_min(1).long() - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, lengths]
        raise ValueError(f"Unsupported pooling={self.pooling!r}")

    def forward(self, input_ids, attention_mask):
        if not hasattr(self.backbone, "transformer"):
            raise TypeError(
                "DNAClassifier in step5_full_finetune_eval.py expects a GPT-2-style "
                "backbone with a .transformer module. Use the architecture-specific "
                "Step 5 wrapper for non-GPT backbones."
            )
        out    = self.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state                         # (B, T, D)
        pooled = self._pool_hidden(hidden, attention_mask)
        # No L2 normalisation here — fine-tuning with a trained head benefits
        # from unconstrained embedding space. (step4 linear probe normalises
        # because logistic regression on the unit sphere is more stable.)
        return self.classifier(pooled)                         # (B, n_classes)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_task(
    base_model,
    tokenizer,
    train_seqs, train_labels,
    test_seqs,  test_labels,
    source:        str,
    device:        str,
    args,
) -> dict:
    """
    Fine-tune a fresh full backbone copy of base_model on one task.
    Returns held-out evaluation metrics after model selection on validation.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split

    # Encode integer labels (handle string labels from GB)
    le      = LabelEncoder()
    train_y = le.fit_transform(train_labels)
    test_y  = le.transform(test_labels)
    n_cls   = len(le.classes_)

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

    # Build datasets
    train_ds = SeqDataset(fit_seqs, fit_y, tokenizer, args.max_length)
    val_ds   = SeqDataset(val_seqs, val_y, tokenizer, args.max_length)
    test_ds  = SeqDataset(test_seqs, test_y, tokenizer, args.max_length)
    pad_id   = tokenizer.pad_token_id
    collate  = lambda b: _collate_fn(b, pad_id)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"),
                          collate_fn=collate, generator=generator)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device == "cuda"),
                          collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device == "cuda"),
                          collate_fn=collate)

    # Fresh full-finetuning copy per task; never modify the shared base_model.
    backbone   = copy.deepcopy(base_model)
    # Gradient checkpointing: recompute activations during backward instead of storing
    # them all simultaneously. ~30% slower per step but prevents OOM on large tasks.
    if args.grad_ckpt:
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable({"use_reentrant": False})
        elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "gradient_checkpointing_enable"):
            backbone.transformer.gradient_checkpointing_enable({"use_reentrant": False})
    hidden_dim = backbone.config.n_embd
    dtype      = next(backbone.parameters()).dtype
    classifier = DNAClassifier(
        backbone,
        n_cls,
        hidden_dim,
        pooling=args.pooling,
        head_type=args.head_type,
        head_dropout=args.head_dropout,
    ).to(device=device, dtype=dtype)

    # Optimizer: all backbone params + classifier head.
    trainable  = [p for p in classifier.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in classifier.parameters())
    print(
        f"    trainable params={n_trainable/1e6:.2f}M / "
        f"total params={n_total/1e6:.2f}M"
    )
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr,
                                   weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(train_dl) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.warmup_steps >= 0:
        warmup = min(args.warmup_steps, max(1, total_steps - 1))
    else:
        warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    monitor_name = args.monitor
    best_metric = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        # ── train ──────────────────────────────────────────────────────────
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_dl, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = classifier(input_ids, attention_mask)
            loss   = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            loss   = loss / args.grad_accum

            loss.backward()
            if step_idx % args.grad_accum == 0 or step_idx == len(train_dl):
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            del input_ids, attention_mask, labels, logits, loss

        avg_loss = total_loss / len(train_dl)

        # ── eval ───────────────────────────────────────────────────────────
        classifier.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        with torch.inference_mode():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                logits = classifier(input_ids, attention_mask)
                val_loss = F.cross_entropy(logits, labels)
                total_val_loss += val_loss.item()
                preds  = logits.argmax(dim=-1)
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

        history.append({
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
        })

        print(f"    epoch {epoch+1:02d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"acc={acc*100:.2f}%  F1={f1*100:.2f}%  "
              f"MCC={mcc*100:.2f}%  lr={current_lr:.2e}  "
              f"[monitor={monitor_name}:{monitor_val*100:.2f}%]"
              + ("  *best" if improved else ""))

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"    early stop at epoch {epoch+1:02d}  "
                  f"(best {monitor_name} at epoch {best_epoch:02d})")
            break

    if best_state is not None:
        classifier.load_state_dict(best_state)

    # Final metrics on the held-out evaluation split:
    # - GB / NT: official test
    # - GUE / GUE+: provided dev
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            logits = classifier(input_ids, attention_mask)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            del input_ids, attention_mask, labels, logits, preds

    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    mcc = float(matthews_corrcoef(all_labels, all_preds))

    # Cleanup VRAM between tasks
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


# ── Model loader ──────────────────────────────────────────────────────────────

def load_base_model(spec: ModelSpec, device: str, dtype):
    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode})")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager",
    )
    if spec.mode == "bidir":
        model = patch_to_bidirectional(model)

    # Keep on CPU — moved to device per-task inside DNAClassifier
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


# ── Results table ─────────────────────────────────────────────────────────────

def _metric_val(task_metrics, metric: str) -> float:
    if isinstance(task_metrics, dict):
        return task_metrics.get(metric, float("nan"))
    return float(task_metrics)


def print_results_table(results: dict, benchmarks: list, title: str, metric: str = "accuracy"):
    short_names = [b[2] for b in benchmarks]
    model_names = list(results.keys())
    col_w  = max(max(len(n) for n in model_names), 6)
    task_w = 13

    print(f"\n{title}  [metric: {metric}]")
    header = (f"{'Model':<{col_w}}  "
              + "  ".join(f"{n:>{task_w}}" for n in short_names)
              + f"  {'Avg':>{task_w}}")
    sep = "=" * len(header)
    print(sep); print(header); print(sep)

    for model_name, task_map in results.items():
        vals = [_metric_val(task_map.get(b[0], {}), metric) for b in benchmarks]
        avg  = np.nanmean(vals)
        row  = f"{model_name:<{col_w}}  "
        row += "  ".join(f"{v*100:>{task_w}.2f}" for v in vals)
        row += f"  {avg*100:>{task_w}.2f}"
        print(row)

    print(sep)
    print(f"({metric} %, full fine-tuning, GB/NT use train→val for selection then test; GUE/GUE+ report dev)")


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DNA-LLM2Vec Step 5: Full Fine-tuning Evaluation")

    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode")
    p.add_argument("--task-keys", nargs="+", default=None,
                   help="Optional subset of task keys to run, e.g. epi_0 epi_1.")

    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--gb-only",       action="store_true")
    suite.add_argument("--nt-only",       action="store_true")
    suite.add_argument("--gue-only",      action="store_true")
    suite.add_argument("--gue-plus-only", action="store_true",
                       help="Run GUE+ EPI only (6 tasks, 5000 bp sequences).")

    p.add_argument("--output",       default="./eval_results/ft_results.json")
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--grad-accum",   type=int,   default=1,
                   help="Number of gradient accumulation steps (default: 1).")
    p.add_argument("--max-length",   type=int,   default=1024)
    p.add_argument("--gb-root", default=os.environ.get("GENOMIC_BENCHMARKS_DIR", os.path.expanduser("~/.genomic_benchmarks")),
                   help="Root directory for genomic_benchmarks task folders. If a task is missing here, it will be downloaded.")
    p.add_argument("--benchmark-cache-dir", default="./cache/benchmarks",
                   help="Directory for serialized benchmark caches (used to avoid repeatedly scanning GB small files).")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Fraction of total steps used for LR warmup when --warmup-steps < 0.")
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="Fixed LR warmup steps, matching DNABERT-2 Table 7 style. "
                        "Set to -1 to use --warmup-ratio instead.")
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--grad-ckpt",    action="store_true",
                   help="Enable gradient checkpointing to reduce activation memory.")
    p.add_argument("--pooling", choices=("average", "last", "weighted"), default="average",
                   help="Sequence pooling used before the classification head.")
    p.add_argument("--head-type", choices=("linear", "mlp"), default="linear",
                   help="Classification head: linear or a small LayerNorm-MLP.")
    p.add_argument("--head-dropout", type=float, default=0.1,
                   help="Dropout used inside --head-type mlp.")
    p.add_argument("--patience",     type=int, default=0,
                   help="Early stopping patience in epochs. 0 disables early stopping.")
    p.add_argument("--monitor",      choices=("f1", "mcc", "accuracy"), default="mcc",
                   help="Validation metric used for best-model selection and early stopping.")
    p.add_argument("--min-delta",    type=float, default=0.0,
                   help="Minimum improvement required to reset early stopping.")
    p.add_argument("--seed",         type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--run-name",     default=None,
                   help="Optional W&B run name. If omitted, a descriptive name is generated.")
    p.add_argument("--repeat-index", type=int, default=None,
                   help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    p.add_argument("--no-wandb",     action="store_true")
    p.add_argument("--gue-plus-dir", default="",
                   help="Root directory of the locally downloaded GUE+ data "
                        "(parent of the EPI/ folder). If provided during the "
                        "default full evaluation, the 6 EPI tasks are appended "
                        "to GB+NT+GUE.")
    p.add_argument("--epi-subdir-names", nargs=6, default=None,
                   metavar=("S0","S1","S2","S3","S4","S5"),
                   help="Override EPI subdir names for epi_0…epi_5 "
                        "(default: GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK).")
    p.add_argument("--epi-crop-bp",  type=int, default=4096,
                   help="Centre-crop EPI sequences to this many bp (default: 4096).")
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="center",
                   help="How to crop GUE+ EPI sequences: center uses the sequence midpoint; "
                        "junction centers on the enhancer/promoter boundary.")
    p.add_argument("--filter-n", action="store_true",
                   help="For GUE+ EPI, drop rows whose cropped sequence contains any non-ACGT base.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    specs = [ModelSpec.parse(s) for s in args.models]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    global _gb_root, _benchmark_cache_dir
    _gb_root = os.path.expanduser(args.gb_root)
    _benchmark_cache_dir = args.benchmark_cache_dir

    global _epi_subdir_map
    if args.epi_subdir_names:
        _epi_subdir_map = dict(zip([f"epi_{i}" for i in range(6)], args.epi_subdir_names))
    else:
        _epi_subdir_map = dict(_EPI_SUBDIR_DEFAULT)

    if args.gb_only:
        active = GB_BENCHMARKS
    elif args.nt_only:
        active = NT_BENCHMARKS
    elif args.gue_only:
        active = GUE_BENCHMARKS
    elif args.gue_plus_only:
        if not args.gue_plus_dir:
            print("ERROR: --gue-plus-only requires --gue-plus-dir <path>")
            sys.exit(1)
        active = GUE_PLUS_EPI_BENCHMARKS
    else:
        active = ALL_BENCHMARKS + (GUE_PLUS_EPI_BENCHMARKS if args.gue_plus_dir else [])

    if args.task_keys:
        keep = set(args.task_keys)
        active = [b for b in active if b[0] in keep]
        missing = sorted(keep - {b[0] for b in active})
        if missing:
            print(f"ERROR: unknown --task-keys for the selected suite: {missing}")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Full fine-tuning in fp32 is usually too memory-heavy for 1024-token EPI,
    # so CUDA runs in bfloat16. Use --grad-ckpt if VRAM is tight.
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    gb_n  = sum(1 for b in active if b[3] == "gb")
    nt_n  = sum(1 for b in active if b[3] == "nt")
    gue_n  = sum(1 for b in active if b[3] == "gue")
    guep_n = sum(1 for b in active if b[3] == "gue+")

    print("=" * 68)
    print("DNA-LLM2Vec  |  Step 5 - Full Fine-tuning Evaluation")
    print("=" * 68)
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  Tasks      : {gb_n} GB  +  {nt_n} NT  +  {gue_n} GUE  +  {guep_n} GUE+ EPI  =  {len(active)} total")
    print(f"  Models     : {[s.name for s in specs]}")
    print(f"  Epochs     : {args.epochs}  |  LR: {args.lr}  |  warmup_steps={args.warmup_steps}")
    print(f"  Early stop : patience={args.patience}  |  monitor={args.monitor}  |  min_delta={args.min_delta}")
    print(f"  Max length : {args.max_length}")
    print(f"  Seed       : {args.seed}")
    if args.repeat_index is not None:
        print(f"  Repeat     : {args.repeat_index}")
    print("=" * 68)

    # ── W&B ───────────────────────────────────────────────────────────────
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "dna_foundation"
        os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"
        try:
            import wandb
            model_label = "_".join(s.name for s in specs)
            suite_label = (
                "gb" if args.gb_only else
                "nt" if args.nt_only else
                "gue" if args.gue_only else
                "gueplus" if args.gue_plus_only else
                "all"
            )
            repeat_suffix = f"_r{args.repeat_index}" if args.repeat_index is not None else ""
            run_name = args.run_name or f"step5_full_{suite_label}_{model_label}_s{args.seed}{repeat_suffix}"
            wandb.init(
                job_type="full_finetune_eval",
                name=run_name,
                config={
                    "models":     args.models,
                    "epochs":     args.epochs,
                    "lr":         args.lr,
                    "max_length": args.max_length,
                    "patience":   args.patience,
                    "monitor":    args.monitor,
                    "min_delta":  args.min_delta,
                    "seed":       args.seed,
                    "repeat_index": args.repeat_index,
                    "gb_tasks":   gb_n,
                    "nt_tasks":   nt_n,
                    "gue_tasks":  gue_n,
                    "gue_plus_epi_tasks": guep_n,
                },
            )
        except Exception as e:
            print(f"  [warn] W&B init failed: {e}")

    # ── Load datasets once (sequences only; tokenisation happens per task) ─
    print("\n[1/3] Loading datasets")
    benchmark_data = {}
    for task_key, _, display_name, source in active:
        print(f"  [{source.upper()}] {display_name} ({task_key})")
        tr_s, tr_l, te_s, te_l = load_benchmark_data(
            task_key, source,
            gue_plus_dir=args.gue_plus_dir,
            crop_bp=args.epi_crop_bp,
            crop_mode=args.epi_crop_mode,
            filter_non_acgt=args.filter_n,
        )
        benchmark_data[task_key] = (tr_s, tr_l, te_s, te_l)
        split_label = "dev" if source in ("gue", "gue+") else "test"
        extra = ""
        if source == "gue+":
            extra = f"  [{args.epi_crop_mode}-cropped to {args.epi_crop_bp} bp"
            extra += ", filter-n" if args.filter_n else ""
            extra += "]"
        print(f"       train={len(tr_s):,}  {split_label}={len(te_s):,}{extra}")

    # ── Evaluate each model ───────────────────────────────────────────────
    print("\n[2/3] Fine-tuning models")
    results = {}

    for spec in specs:
        print(f"\n── {spec.name} ──────────────────────────────────────────────")
        base_model, tokenizer = load_base_model(spec, device, dtype)
        # Keep the shared base on CPU. Each task deep-copies it, then moves
        # only the active classifier/backbone to GPU. This avoids
        # keeping an extra full backbone resident in VRAM during fine-tuning.
        results[spec.name] = {}

        for task_key, n_classes, display_name, source in active:
            tr_s, tr_l, te_s, te_l = benchmark_data[task_key]
            print(f"\n  [{source.upper()}] {display_name}  "
                  f"(train={len(tr_s):,}, n_classes={n_classes})")

            metrics = train_one_task(
                base_model, tokenizer,
                tr_s, tr_l, te_s, te_l,
                source=source,
                device=device,
                args=args,
            )
            results[spec.name][task_key] = metrics

            if source in ("gue", "gue+"):
                print(f"  ✓ {display_name:<28}  "
                      f"F1={metrics['f1']*100:.2f}%  MCC={metrics['mcc']*100:.2f}%")
            else:
                print(
                    f"  ✓ {display_name:<28}  acc={metrics['accuracy']*100:.2f}%  "
                    f"F1={metrics['f1']*100:.2f}%  MCC={metrics['mcc']*100:.2f}%"
                )

        # Free base model VRAM before loading the next one
        del base_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ── Aggregate averages ────────────────────────────────────────────────
    print("\n[3/3] Results")

    gb_active   = [b for b in active if b[3] == "gb"]
    nt_active   = [b for b in active if b[3] == "nt"]
    gue_active  = [b for b in active if b[3] == "gue"]
    guep_active = [b for b in active if b[3] == "gue+"]

    for model_name, task_map in results.items():
        if gb_active:
            results[model_name]["avg_gb_accuracy"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "accuracy") for b in gb_active]
            ))
            results[model_name]["avg_gb_f1"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "f1") for b in gb_active]
            ))
            results[model_name]["avg_gb_mcc"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "mcc") for b in gb_active]
            ))
        if nt_active:
            results[model_name]["avg_nt_accuracy"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "accuracy") for b in nt_active]
            ))
            results[model_name]["avg_nt_f1"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "f1") for b in nt_active]
            ))
            results[model_name]["avg_nt_mcc"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "mcc") for b in nt_active]
            ))
        if gue_active:
            results[model_name]["avg_gue_f1"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "f1") for b in gue_active]
            ))
            results[model_name]["avg_gue_mcc"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "mcc") for b in gue_active]
            ))
        if guep_active:
            results[model_name]["avg_epi_f1"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "f1") for b in guep_active]
            ))
            results[model_name]["avg_epi_mcc"] = float(np.nanmean(
                [_metric_val(task_map.get(b[0], {}), "mcc") for b in guep_active]
            ))

    if gb_active:
        print_results_table(results, gb_active,
                            "Genomics Benchmarks (Grevsova et al., 2023)",
                            metric="accuracy")
        print_results_table(results, gb_active,
                            "Genomics Benchmarks (Grevsova et al., 2023)",
                            metric="f1")
        print_results_table(results, gb_active,
                            "Genomics Benchmarks (Grevsova et al., 2023)",
                            metric="mcc")
    if nt_active:
        print_results_table(results, nt_active,
                            "NT Downstream Tasks (Dalla-Torre et al., 2023)",
                            metric="accuracy")
        print_results_table(results, nt_active,
                            "NT Downstream Tasks (Dalla-Torre et al., 2023)",
                            metric="f1")
        print_results_table(results, nt_active,
                            "NT Downstream Tasks (Dalla-Torre et al., 2023)",
                            metric="mcc")
    if gue_active:
        print_results_table(results, gue_active,
                            "GUE — Genome Understanding Evaluation (DNABERT-2)",
                            metric="f1")
        print_results_table(results, gue_active,
                            "GUE — Genome Understanding Evaluation (DNABERT-2)",
                            metric="mcc")
    if guep_active:
        crop_note = f"{args.epi_crop_mode}-cropped to {args.epi_crop_bp} bp"
        print_results_table(results, guep_active,
                            f"GUE+ EPI — Enhancer-Promoter Interaction ({crop_note})",
                            metric="f1")
        print_results_table(results, guep_active,
                            f"GUE+ EPI — Enhancer-Promoter Interaction ({crop_note})",
                            metric="mcc")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    print("\nSummary averages:")
    for model_name in results:
        m = results[model_name]
        parts = []
        if "avg_gb_accuracy" in m:
            parts.append(f"GB acc={m['avg_gb_accuracy']*100:.2f}%")
        if "avg_gb_f1" in m:
            parts.append(f"GB F1={m['avg_gb_f1']*100:.2f}%")
        if "avg_gb_mcc" in m:
            parts.append(f"GB MCC={m['avg_gb_mcc']*100:.2f}%")
        if "avg_nt_accuracy" in m:
            parts.append(f"NT acc={m['avg_nt_accuracy']*100:.2f}%")
        if "avg_nt_f1" in m:
            parts.append(f"NT F1={m['avg_nt_f1']*100:.2f}%")
        if "avg_nt_mcc" in m:
            parts.append(f"NT MCC={m['avg_nt_mcc']*100:.2f}%")
        if "avg_gue_f1" in m:
            parts.append(f"GUE F1={m['avg_gue_f1']*100:.2f}%")
        if "avg_gue_mcc" in m:
            parts.append(f"GUE MCC={m['avg_gue_mcc']*100:.2f}%")
        if "avg_epi_f1" in m:
            parts.append(f"EPI F1={m['avg_epi_f1']*100:.2f}%")
        if "avg_epi_mcc" in m:
            parts.append(f"EPI MCC={m['avg_epi_mcc']*100:.2f}%")
        print(f"  {model_name}: " + "  |  ".join(parts))

    if not args.no_wandb:
        try:
            import wandb
            if wandb.run is not None:
                flat = {}
                for m, task_map in results.items():
                    for t, v in task_map.items():
                        if isinstance(v, dict):
                            for mn, mv in v.items():
                                flat[f"{m}/{t}/{mn}"] = mv
                        else:
                            flat[f"{m}/{t}"] = v
                wandb.log(flat)
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
