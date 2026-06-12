"""
DNALONGBENCH ETGP training/evaluation for HyenaDNA.

This is intentionally separate from Step 4 because ETGP uses 450 kb inputs and
has different data/metric assumptions from the short-sequence linear probes.

Supported manifests:
  1. sequence,label,split
  2. chrom,start,end,label,split with --genome-fasta

Labels may be 0/1, negative/positive, false/true, or no/yes. Splits should be
train/valid/test (dev/val are accepted as valid).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
CURRENT_DIR = os.path.dirname(__file__)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from common import extract_hidden_states, load_hyena_backbone, load_hyena_tokenizer  # noqa: E402


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1].upper()


def normalize_split(value: str) -> str:
    split = value.strip().lower()
    if split in {"val", "dev", "valid", "validation"}:
        return "valid"
    if split in {"train", "training"}:
        return "train"
    if split in {"test", "testing"}:
        return "test"
    raise ValueError(f"Unrecognized split value: {value!r}")


def parse_label(value: str) -> int:
    label = str(value).strip().lower()
    if label in {"1", "true", "t", "yes", "y", "pos", "positive"}:
        return 1
    if label in {"0", "false", "f", "no", "n", "neg", "negative"}:
        return 0
    try:
        numeric = float(label)
    except ValueError as e:
        raise ValueError(f"Unrecognized binary label: {value!r}") from e
    if numeric in (0.0, 1.0):
        return int(numeric)
    raise ValueError(f"Binary label must be 0/1, got: {value!r}")


def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def sniff_dialect(path: str) -> csv.Dialect:
    with open_text(path) as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        return csv.excel_tab if path.endswith((".tsv", ".tsv.gz")) else csv.excel


def read_fasta(path: str) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    name: Optional[str] = None
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                records[name] = []
            elif name is not None:
                records[name].append(line.upper())
    return {key: "".join(parts) for key, parts in records.items()}


@dataclass
class ETGPRecord:
    sequence: str
    label: int
    split: str


def _get_first(row: dict[str, str], names: Iterable[str]) -> Optional[str]:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and value != "":
            return value
    return None


def load_manifest(
    path: str,
    genome: Optional[dict[str, str]],
    sequence_column: str,
    label_column: str,
    split_column: str,
    chrom_column: str,
    start_column: str,
    end_column: str,
    strand_column: str,
    max_length: int,
    pad_to_length: bool,
) -> list[ETGPRecord]:
    dialect = sniff_dialect(path)
    records: list[ETGPRecord] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")

        for row in reader:
            label_raw = _get_first(row, [label_column, "target", "y", "class", "positive"])
            split_raw = _get_first(row, [split_column, "partition", "set"])
            if label_raw is None or split_raw is None:
                raise ValueError(
                    f"Manifest row is missing label/split columns. Available columns: {reader.fieldnames}"
                )

            seq = _get_first(row, [sequence_column, "seq", "dna", "input"])
            if seq is None:
                if genome is None:
                    raise ValueError("Manifest has no sequence column; provide --genome-fasta for coordinate rows.")
                chrom = _get_first(row, [chrom_column, "chr", "chromosome"])
                start_raw = _get_first(row, [start_column, "begin"])
                end_raw = _get_first(row, [end_column, "stop"])
                if chrom is None or start_raw is None or end_raw is None:
                    raise ValueError(
                        "Coordinate manifest requires chrom/start/end columns or explicit column arguments."
                    )
                chrom_seq = genome.get(chrom) or genome.get(chrom.removeprefix("chr")) or genome.get(f"chr{chrom}")
                if chrom_seq is None:
                    raise KeyError(f"Chromosome {chrom!r} not found in --genome-fasta.")
                start = int(float(start_raw))
                end = int(float(end_raw))
                seq = chrom_seq[start:end]
                strand = _get_first(row, [strand_column, "direction"])
                if strand == "-":
                    seq = reverse_complement(seq)

            seq = seq.strip().upper()
            if pad_to_length and len(seq) < max_length:
                seq = seq + ("N" * (max_length - len(seq)))
            if len(seq) > max_length:
                seq = seq[:max_length]

            records.append(
                ETGPRecord(
                    sequence=seq,
                    label=parse_label(label_raw),
                    split=normalize_split(split_raw),
                )
            )
    return records


class ETGPDataset(Dataset):
    def __init__(self, records: list[ETGPRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ETGPRecord:
        return self.records[idx]


class ETGPCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[ETGPRecord]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            [item.sequence for item in batch],
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                attention_mask = torch.ones_like(enc["input_ids"], dtype=torch.long)
            else:
                attention_mask = enc["input_ids"].ne(int(pad_id)).long()
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": attention_mask,
            "labels": torch.tensor([item.label for item in batch], dtype=torch.float32),
        }


class HyenaETGPClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.to(dtype=next(self.classifier.parameters()).dtype)
        return self.classifier(pooled).squeeze(-1)


def infer_hidden_dim(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("d_model", "hidden_size", "n_embd"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None and hasattr(embeddings, "weight"):
        return int(embeddings.weight.shape[1])
    raise RuntimeError("Could not infer hidden dimension.")


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float((preds == labels).mean()),
    }
    try:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
        metrics["f1"] = float(f1_score(labels, preds))
    except Exception as e:
        print(f"  sklearn metrics unavailable or undefined: {e}")
    return metrics


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: str, desc: str) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    labels_all = []
    logits_all = []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        losses.append(float(loss.detach().cpu()))
        labels_all.append(labels.detach().cpu().numpy())
        logits_all.append(logits.detach().float().cpu().numpy())
    labels_np = np.concatenate(labels_all).astype(np.int64)
    logits_np = np.concatenate(logits_all)
    return float(np.mean(losses)), compute_metrics(labels_np, logits_np)


def split_records(records: list[ETGPRecord]) -> tuple[list[ETGPRecord], list[ETGPRecord], list[ETGPRecord]]:
    train = [record for record in records if record.split == "train"]
    valid = [record for record in records if record.split == "valid"]
    test = [record for record in records if record.split == "test"]
    if not train or not test:
        raise ValueError(f"Need non-empty train/test splits, got train={len(train)}, valid={len(valid)}, test={len(test)}")
    if not valid:
        print("  No valid split found; using test split for validation metrics.")
        valid = test
    return train, valid, test


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(trainable)


def main():
    parser = argparse.ArgumentParser(description="DNALONGBENCH ETGP HyenaDNA probe/fine-tuning")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, help="CSV/TSV manifest with sequences or genomic coordinates.")
    parser.add_argument("--genome-fasta", default=None, help="Genome FASTA for coordinate manifests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=450000)
    parser.add_argument("--pad-to-length", action="store_true")

    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--chrom-column", default="chrom")
    parser.add_argument("--start-column", default="start")
    parser.add_argument("--end-column", default="end")
    parser.add_argument("--strand-column", default="strand")

    parser.add_argument("--full-finetune", action="store_true", help="Train backbone too; default is frozen linear probe.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 72)
    print("DNALONGBENCH ETGP | HyenaDNA")
    print("=" * 72)
    print(f"  Model                    : {args.model}")
    print(f"  Manifest                 : {args.manifest}")
    print(f"  Output                   : {args.output}")
    print(f"  Device / dtype           : {device} / {dtype}")
    print(f"  Max length               : {args.max_length:,}")
    print(f"  Train mode               : {'full fine-tune' if args.full_finetune else 'frozen linear probe'}")

    genome = read_fasta(args.genome_fasta) if args.genome_fasta else None
    records = load_manifest(
        path=args.manifest,
        genome=genome,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        split_column=args.split_column,
        chrom_column=args.chrom_column,
        start_column=args.start_column,
        end_column=args.end_column,
        strand_column=args.strand_column,
        max_length=args.max_length,
        pad_to_length=args.pad_to_length,
    )
    train_records, valid_records, test_records = split_records(records)
    print(f"  Samples                  : train={len(train_records):,}, valid={len(valid_records):,}, test={len(test_records):,}")
    print(
        "  Positive rate            : "
        f"train={np.mean([r.label for r in train_records]):.3f}, "
        f"valid={np.mean([r.label for r in valid_records]):.3f}, "
        f"test={np.mean([r.label for r in test_records]):.3f}"
    )
    lengths = np.array([len(r.sequence) for r in records])
    print(f"  Lengths                  : min={lengths.min():,}, median={int(np.median(lengths)):,}, max={lengths.max():,}")

    tokenizer = load_hyena_tokenizer(args.model)
    backbone, load_path = load_hyena_backbone(args.model, device=device, dtype=dtype)
    print(f"  Load path                : {load_path}")
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    set_backbone_trainable(backbone, args.full_finetune)

    model = HyenaETGPClassifier(
        backbone=backbone,
        hidden_dim=infer_hidden_dim(backbone),
        dropout=args.head_dropout,
    ).to(device)

    collator = ETGPCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        ETGPDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    valid_loader = DataLoader(
        ETGPDataset(valid_records),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        ETGPDataset(test_records),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters     : {trainable / 1e6:.3f}M / {total / 1e6:.3f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    best_valid = -float("inf")
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", step=global_step)

                if global_step >= total_steps:
                    break
        valid_loss, valid_metrics = evaluate(model, valid_loader, device, desc="valid")
        score = valid_metrics.get("auroc", -valid_loss)
        print(f"  Epoch {epoch + 1} valid loss={valid_loss:.4f} metrics={valid_metrics}")
        if score > best_valid:
            best_valid = score
            torch.save(
                {
                    "classifier": model.classifier.state_dict(),
                    "args": vars(args),
                    "valid_metrics": valid_metrics,
                    "valid_loss": valid_loss,
                },
                os.path.join(args.output, "best_head.pt"),
            )
            if args.full_finetune:
                torch.save(model.state_dict(), os.path.join(args.output, "best_model_state.pt"))
        if global_step >= total_steps:
            break

    test_loss, test_metrics = evaluate(model, test_loader, device, desc="test")
    print(f"  Test loss                : {test_loss:.4f}")
    print(f"  Test metrics             : {test_metrics}")
    with open(os.path.join(args.output, "metrics.tsv"), "w", encoding="utf-8") as handle:
        handle.write("split\tmetric\tvalue\n")
        for key, value in test_metrics.items():
            handle.write(f"test\t{key}\t{value}\n")
        handle.write(f"test\tloss\t{test_loss}\n")


if __name__ == "__main__":
    main()
