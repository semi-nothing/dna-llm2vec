"""
DNALONGBENCH RSAP training/evaluation for HyenaDNA.

RSAP is handled as regulatory sequence activity regression from long genomic windows.
Supported manifests:
  1. sequence,split,<one or more activity label columns>
  2. chrom,start,end,split,<one or more activity label columns>
     with --genome-fasta

Metrics are reported both as macro means across activity targets and, for
multi-output manifests, per target.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
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
METADATA_COLUMNS = {
    "sequence",
    "seq",
    "dna",
    "input",
    "chrom",
    "chr",
    "chromosome",
    "start",
    "begin",
    "end",
    "stop",
    "split",
    "partition",
    "set",
    "strand",
    "direction",
    "gene_id",
    "gene_name",
    "gene_tss",
    "tss",
}


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


def _get_first(row: dict[str, str], names: Iterable[str]) -> Optional[str]:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and value != "":
            return value
    return None


def parse_float(value: str) -> float | None:
    value = str(value).strip()
    if value == "" or value.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def infer_label_columns(fieldnames: list[str]) -> list[str]:
    labels = [name for name in fieldnames if name.lower() not in METADATA_COLUMNS]
    if not labels:
        raise ValueError("Could not infer activity label columns. Pass --label-columns explicitly.")
    return labels


@dataclass
class RSAPRecord:
    sequence: str
    labels: np.ndarray
    split: str


def load_manifest(
    path: str,
    genome: Optional[dict[str, str]],
    label_columns: Optional[list[str]],
    sequence_column: str,
    split_column: str,
    chrom_column: str,
    start_column: str,
    end_column: str,
    strand_column: str,
    max_length: int,
    pad_to_length: bool,
) -> tuple[list[RSAPRecord], list[str]]:
    dialect = sniff_dialect(path)
    records: list[RSAPRecord] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        labels = label_columns or infer_label_columns(reader.fieldnames)

        skipped = 0
        for row in reader:
            split_raw = _get_first(row, [split_column, "partition", "set"])
            if split_raw is None:
                raise ValueError(f"Manifest row is missing split column. Available columns: {reader.fieldnames}")

            parsed_labels = [parse_float(row.get(name, "")) for name in labels]
            if any(value is None for value in parsed_labels):
                skipped += 1
                continue

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
                RSAPRecord(
                    sequence=seq,
                    labels=np.asarray(parsed_labels, dtype=np.float32),
                    split=normalize_split(split_raw),
                )
            )
    if skipped:
        print(f"  Skipped rows with missing labels: {skipped:,}")
    return records, labels


class RSAPDataset(Dataset):
    def __init__(self, records: list[RSAPRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> RSAPRecord:
        return self.records[idx]


class RSAPCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[RSAPRecord]) -> dict[str, torch.Tensor]:
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
            "labels": torch.tensor(np.stack([item.labels for item in batch]), dtype=torch.float32),
        }


class HyenaRSAPRegressor(nn.Module):
    def __init__(self, backbone, hidden_dim: int, n_targets: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
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
        pooled = pooled.to(dtype=next(self.regressor.parameters()).dtype)
        return self.regressor(pooled)


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


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(labels: np.ndarray, preds: np.ndarray, target_names: list[str]) -> dict[str, float]:
    residual = preds - labels
    mse_targets = np.mean(residual**2, axis=0)
    mae_targets = np.mean(np.abs(residual), axis=0)
    var_targets = np.var(labels, axis=0)
    r2_targets = np.where(var_targets > 0, 1.0 - mse_targets / var_targets, np.nan)

    pearson_targets = []
    spearman_targets = []
    for idx in range(labels.shape[1]):
        pearson_targets.append(_corr(labels[:, idx], preds[:, idx]))
        spearman_targets.append(_corr(_rankdata(labels[:, idx]), _rankdata(preds[:, idx])))

    metrics = {
        "mse": float(np.nanmean(mse_targets)),
        "mae": float(np.nanmean(mae_targets)),
        "r2": float(np.nanmean(r2_targets)),
        "pearson": float(np.nanmean(pearson_targets)),
        "spearman": float(np.nanmean(spearman_targets)),
    }
    for idx, name in enumerate(target_names):
        safe_name = name.replace("\t", " ").replace("\n", " ")
        metrics[f"{safe_name}/mse"] = float(mse_targets[idx])
        metrics[f"{safe_name}/mae"] = float(mae_targets[idx])
        metrics[f"{safe_name}/r2"] = float(r2_targets[idx])
        metrics[f"{safe_name}/pearson"] = float(pearson_targets[idx])
        metrics[f"{safe_name}/spearman"] = float(spearman_targets[idx])
    return metrics


@torch.inference_mode()
def evaluate(
    model,
    loader: DataLoader,
    device: str,
    target_names: list[str],
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    desc: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    labels_all = []
    preds_all = []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels_z = (labels - label_mean) / label_std
        preds_z = model(input_ids, attention_mask)
        loss = F.mse_loss(preds_z.float(), labels_z.float())
        losses.append(float(loss.detach().cpu()))
        labels_all.append(labels.detach().cpu().numpy())
        preds_all.append((preds_z * label_std + label_mean).detach().float().cpu().numpy())
    labels_np = np.concatenate(labels_all).astype(np.float32)
    preds_np = np.concatenate(preds_all).astype(np.float32)
    return float(np.mean(losses)), compute_metrics(labels_np, preds_np, target_names)


def split_records(records: list[RSAPRecord]) -> tuple[list[RSAPRecord], list[RSAPRecord], list[RSAPRecord]]:
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
    parser = argparse.ArgumentParser(description="DNALONGBENCH RSAP HyenaDNA probe/fine-tuning")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, help="CSV/TSV manifest with sequences or genomic coordinates.")
    parser.add_argument("--genome-fasta", default=None, help="Genome FASTA for coordinate manifests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=450000)
    parser.add_argument("--pad-to-length", action="store_true")

    parser.add_argument("--label-columns", nargs="+", default=None)
    parser.add_argument("--sequence-column", default="sequence")
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
    print("DNALONGBENCH RSAP | HyenaDNA")
    print("=" * 72)
    print(f"  Model                    : {args.model}")
    print(f"  Manifest                 : {args.manifest}")
    print(f"  Output                   : {args.output}")
    print(f"  Device / dtype           : {device} / {dtype}")
    print(f"  Max length               : {args.max_length:,}")
    print(f"  Train mode               : {'full fine-tune' if args.full_finetune else 'frozen linear probe'}")

    genome = read_fasta(args.genome_fasta) if args.genome_fasta else None
    records, target_names = load_manifest(
        path=args.manifest,
        genome=genome,
        label_columns=args.label_columns,
        sequence_column=args.sequence_column,
        split_column=args.split_column,
        chrom_column=args.chrom_column,
        start_column=args.start_column,
        end_column=args.end_column,
        strand_column=args.strand_column,
        max_length=args.max_length,
        pad_to_length=args.pad_to_length,
    )
    train_records, valid_records, test_records = split_records(records)
    train_labels = np.stack([record.labels for record in train_records]).astype(np.float32)
    label_mean_np = train_labels.mean(axis=0)
    label_std_np = train_labels.std(axis=0)
    label_std_np = np.where(label_std_np > 1e-6, label_std_np, 1.0).astype(np.float32)
    label_mean = torch.tensor(label_mean_np, dtype=torch.float32, device=device)
    label_std = torch.tensor(label_std_np, dtype=torch.float32, device=device)

    print(f"  Samples                  : train={len(train_records):,}, valid={len(valid_records):,}, test={len(test_records):,}")
    print(f"  Activity targets         : {target_names}")
    lengths = np.array([len(r.sequence) for r in records])
    print(f"  Lengths                  : min={lengths.min():,}, median={int(np.median(lengths)):,}, max={lengths.max():,}")

    tokenizer = load_hyena_tokenizer(args.model)
    backbone, load_path = load_hyena_backbone(args.model, device=device, dtype=dtype)
    print(f"  Load path                : {load_path}")
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    set_backbone_trainable(backbone, args.full_finetune)

    model = HyenaRSAPRegressor(
        backbone=backbone,
        hidden_dim=infer_hidden_dim(backbone),
        n_targets=len(target_names),
        dropout=args.head_dropout,
    ).to(device)

    collator = RSAPCollator(tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        RSAPDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    valid_loader = DataLoader(
        RSAPDataset(valid_records),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        RSAPDataset(test_records),
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
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        if not args.full_finetune:
            # Frozen linear probe: keep the backbone in eval mode so its
            # dropout does not perturb the (frozen) features during head training.
            model.backbone.eval()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels_z = (labels - label_mean) / label_std
            preds_z = model(input_ids, attention_mask)
            loss = F.mse_loss(preds_z.float(), labels_z.float())
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
        valid_loss, valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            target_names,
            label_mean,
            label_std,
            desc="valid",
        )
        score = valid_metrics.get("pearson", -valid_loss)
        print(f"  Epoch {epoch + 1} valid loss={valid_loss:.4f} metrics={valid_metrics}")
        if score > best_valid:
            best_valid = score
            torch.save(
                {
                    "regressor": model.regressor.state_dict(),
                    "args": vars(args),
                    "target_names": target_names,
                    "label_mean": label_mean_np,
                    "label_std": label_std_np,
                    "valid_metrics": valid_metrics,
                    "valid_loss": valid_loss,
                },
                os.path.join(args.output, "best_head.pt"),
            )
            if args.full_finetune:
                torch.save(model.state_dict(), os.path.join(args.output, "best_model_state.pt"))
        if global_step >= total_steps:
            break

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        target_names,
        label_mean,
        label_std,
        desc="test",
    )
    print(f"  Test loss                : {test_loss:.4f}")
    print(f"  Test metrics             : {test_metrics}")
    with open(os.path.join(args.output, "metrics.tsv"), "w", encoding="utf-8") as handle:
        handle.write("split\tmetric\tvalue\n")
        for key, value in test_metrics.items():
            handle.write(f"test\t{key}\t{value}\n")
        handle.write(f"test\tloss\t{test_loss}\n")


if __name__ == "__main__":
    main()
