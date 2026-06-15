"""
1M HyenaDNA Step 3 with explicit pre-sampled negatives.

This is a separate long-context entry point and does not change the existing
short-context Step 3 scripts. It reads long-window coordinates plus a negative
manifest, fetches sequence from a reference FASTA on demand, and trains with an
explicit-negative InfoNCE loss so very small physical batches remain usable.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import random
import sys
import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, set_seed


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from common import (  # noqa: E402
    count_trainable_parameters_m,
    extract_hidden_states,
    load_hyena_causal_lm,
    load_hyena_tokenizer,
    save_hyena_checkpoint,
)
from hyena_bidirectional import inspect_hyenadna_bidirectional, make_hyenadna_bidirectional  # noqa: E402
from step3_hyena_contrastive_lora import (  # noqa: E402
    _attention_mask,
    _peft_base_model,
    apply_hyena_lora,
    promote_trainable_parameters_to_fp32,
    reverse_complement,
    set_dropout,
)


DEFAULT_HYENA_1M_MODEL = "LongSafari/hyenadna-large-1m-seqlen-hf"


def read_windows_manifest(path: str, split: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "chrom", "start", "end", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] == split:
                rows[row["id"]] = row
    if not rows:
        raise ValueError(f"No windows found for split={split!r} in {path}")
    return rows


def read_negative_manifest(path: str, split: str, rows_by_id: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    examples = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "anchor_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an anchor_id column.")
        neg_columns = [name for name in reader.fieldnames if name.startswith("neg_id_")]
        if not neg_columns:
            raise ValueError(f"{path} must contain at least one neg_id_* column.")

        for row in reader:
            if row.get("split", split) != split:
                continue
            anchor_id = row["anchor_id"]
            if anchor_id not in rows_by_id:
                continue
            neg_ids = [
                row[col]
                for col in neg_columns
                if row.get(col) and row[col] in rows_by_id and row[col] != anchor_id
            ]
            if neg_ids:
                examples.append({"anchor_id": anchor_id, "neg_ids": neg_ids})

    if not examples:
        raise ValueError(f"No usable negative rows found for split={split!r} in {path}")
    return examples


class FastaWindowStore:
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self._fasta = None

    @property
    def fasta(self):
        if self._fasta is None:
            from pyfaidx import Fasta

            self._fasta = Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        return self._fasta

    def fetch(self, row: dict[str, str]) -> str:
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        return str(self.fasta[chrom][start:end]).upper()


class LongWindowExplicitNegativeDataset(TorchDataset):
    def __init__(
        self,
        fasta_path: str,
        rows_by_id: dict[str, dict[str, str]],
        examples: list[dict[str, object]],
        negatives_per_anchor: int,
        negative_choice: str,
        seed: int,
    ):
        if negatives_per_anchor <= 0:
            raise ValueError("negatives_per_anchor must be positive.")
        self.store = FastaWindowStore(fasta_path)
        self.rows_by_id = rows_by_id
        self.examples = examples
        self.negatives_per_anchor = negatives_per_anchor
        self.negative_choice = negative_choice
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.examples)

    def _choose_negative_ids(self, neg_ids: list[str]) -> list[str]:
        k = min(self.negatives_per_anchor, len(neg_ids))
        if self.negative_choice == "first":
            return neg_ids[:k]
        return self.rng.sample(neg_ids, k)

    def __getitem__(self, idx):
        example = self.examples[idx]
        anchor_id = str(example["anchor_id"])
        neg_ids = self._choose_negative_ids(list(example["neg_ids"]))
        return {
            "sequence": self.store.fetch(self.rows_by_id[anchor_id]),
            "neg_sequences": [self.store.fetch(self.rows_by_id[neg_id]) for neg_id in neg_ids],
        }


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyenaDNAExplicitNegativeContrastive(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(self, base_model, proj_dim: int = 256, temperature: float = 0.05):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.temperature = temperature
        hidden_size = getattr(self.config, "d_model", None) or getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = int(getattr(base_model.get_input_embeddings(), "embedding_dim"))
        self.proj = ProjectionHead(hidden_size, proj_dim) if proj_dim > 0 else None

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        base = _peft_base_model(self.base_model)
        encoder = base.hyena if hasattr(base, "hyena") else base
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)
        pooled = self._mean_pool(hidden, attention_mask)
        if self.proj is not None:
            pooled = pooled.to(dtype=next(self.proj.parameters()).dtype)
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)

    def explicit_negative_loss(self, z_a: torch.Tensor, z_b: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        # z_neg is [batch, n_neg, dim]. Class 0 is the positive pair.
        pos_ab = (z_a * z_b).sum(dim=-1, keepdim=True)
        neg_a = torch.einsum("bd,bkd->bk", z_a, z_neg)
        logits_a = torch.cat([pos_ab, neg_a], dim=1) / self.temperature

        pos_ba = (z_b * z_a).sum(dim=-1, keepdim=True)
        neg_b = torch.einsum("bd,bkd->bk", z_b, z_neg)
        logits_b = torch.cat([pos_ba, neg_b], dim=1) / self.temperature

        labels = torch.zeros(z_a.size(0), dtype=torch.long, device=z_a.device)
        return (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)) / 2.0

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        input_ids_neg: torch.Tensor,
        attention_mask_neg: torch.Tensor,
        **kwargs,
    ):
        z_a = self.encode(input_ids_a, attention_mask_a)
        z_b = self.encode(input_ids_b, attention_mask_b)
        batch_size, n_neg, seq_len = input_ids_neg.shape
        z_neg = self.encode(
            input_ids_neg.reshape(batch_size * n_neg, seq_len),
            attention_mask_neg.reshape(batch_size * n_neg, seq_len),
        ).reshape(batch_size, n_neg, -1)
        loss = self.explicit_negative_loss(z_a, z_b, z_neg)
        return type("ContrastiveOutput", (), {"loss": loss, "z_a": z_a, "z_b": z_b})()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.base_model.gradient_checkpointing_enable()
            else:
                self.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = self.base_model
        if hasattr(model_to_save, "merge_and_unload"):
            print("  Saving merged LoRA checkpoint")
            model_to_save = model_to_save.merge_and_unload()
        save_hyena_checkpoint(_peft_base_model(model_to_save), None, save_dir)
        if self.proj is not None:
            torch.save(self.proj.state_dict(), os.path.join(save_dir, "contrastive_head.pt"))


@dataclass
class ExplicitNegativeCollator:
    tokenizer: object
    mode: str
    max_length: int
    chunk_size: int
    overlap_ratio: float
    local_shift_ratio: float

    def __post_init__(self):
        self.crop_shift = int(round(self.chunk_size * (1.0 - self.overlap_ratio)))
        self.local_shift = int(round(self.chunk_size * self.local_shift_ratio))

    def _crop_pair(self, seq: str) -> tuple[str, str]:
        required = self.chunk_size + self.crop_shift
        if self.crop_shift <= 0 or len(seq) < required:
            raise ValueError(
                "crop mode needs parent windows longer than each crop: "
                f"len(seq)={len(seq):,}, chunk_size={self.chunk_size:,}, "
                f"required>={required:,} for overlap_ratio={self.overlap_ratio}."
            )
        window_start = random.randint(0, len(seq) - required)
        start_a = window_start + random.randint(0, self.crop_shift)
        start_b = window_start + random.randint(0, self.crop_shift)
        return seq[start_a : start_a + self.chunk_size], seq[start_b : start_b + self.chunk_size]

    def _local_shift_pair(self, seq: str) -> tuple[str, str]:
        required = self.chunk_size + 2 * self.local_shift
        if self.local_shift <= 0 or len(seq) < required:
            raise ValueError(
                "local_shift mode needs parent windows with shift margin: "
                f"len(seq)={len(seq):,}, chunk_size={self.chunk_size:,}, "
                f"required>={required:,} for local_shift_ratio={self.local_shift_ratio}."
            )
        delta = random.randint(-self.local_shift, self.local_shift)
        start_a = random.randint(self.local_shift, len(seq) - self.chunk_size - self.local_shift)
        start_b = start_a + delta
        return seq[start_a : start_a + self.chunk_size], seq[start_b : start_b + self.chunk_size]

    def _make_pair(self, seq: str) -> tuple[str, str]:
        seq = seq.upper()
        if self.mode == "revcomp":
            return seq, reverse_complement(seq)
        if self.mode == "crop":
            return self._crop_pair(seq)
        if self.mode == "local_shift":
            return self._local_shift_pair(seq)
        return seq, seq

    def _negative_view(self, seq: str) -> str:
        seq = seq.upper()
        if self.mode in {"crop", "local_shift"} and len(seq) > self.chunk_size:
            max_start = len(seq) - self.chunk_size
            start = random.randint(0, max_start)
            return seq[start : start + self.chunk_size]
        return seq

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs_a, seqs_b, neg_seqs = [], [], []
        n_neg = len(features[0]["neg_sequences"])
        if n_neg <= 0:
            raise ValueError("Each feature must contain at least one negative sequence.")

        for feature in features:
            if len(feature["neg_sequences"]) != n_neg:
                raise ValueError("All examples in a batch must have the same number of negatives.")
            a, b = self._make_pair(feature["sequence"])
            seqs_a.append(a)
            seqs_b.append(b)
            neg_seqs.extend(self._negative_view(seq) for seq in feature["neg_sequences"])

        enc_a = self.tokenizer(
            seqs_a,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            seqs_b,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        enc_neg = self.tokenizer(
            neg_seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        batch_size = len(features)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": _attention_mask(self.tokenizer, enc_a),
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": _attention_mask(self.tokenizer, enc_b),
            "input_ids_neg": enc_neg["input_ids"].reshape(batch_size, n_neg, -1),
            "attention_mask_neg": _attention_mask(self.tokenizer, enc_neg).reshape(batch_size, n_neg, -1),
        }


class ExplicitNegativeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None and hasattr(processing_class, "save_pretrained"):
            processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def build_training_args(args) -> TrainingArguments:
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kwargs = dict(
        output_dir=os.path.join(args.output, "trainer_state"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=None if args.save_total_limit <= 0 else args.save_total_limit,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        load_best_model_at_end=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available() and not args.no_pin_memory,
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to=[] if args.no_wandb else ["wandb"],
        seed=args.seed,
        run_name=args.run_name or f"hyena_1m_step3_{args.mode}_explicitneg_s{args.seed}",
        bf16=bf16_ok,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "no"
    if "save_safetensors" in params:
        kwargs["save_safetensors"] = False
    return TrainingArguments(**kwargs)


def required_parent_length(args) -> int:
    if args.mode == "crop":
        crop_shift = int(round(args.chunk_size * (1.0 - args.overlap_ratio)))
        return args.chunk_size + crop_shift
    if args.mode == "local_shift":
        local_shift = int(round(args.chunk_size * args.local_shift_ratio))
        return args.chunk_size + 2 * local_shift
    return args.chunk_size


def parse_args():
    p = argparse.ArgumentParser(description="1M HyenaDNA Step 3 with explicit negatives")
    p.add_argument("--model", default=DEFAULT_HYENA_1M_MODEL)
    p.add_argument("--output", default="./hyena_1m_h5_crop_explicitneg")
    p.add_argument("--genome-fasta", required=True)
    p.add_argument("--windows-manifest", required=True)
    p.add_argument("--negative-manifest", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--mode", choices=("dropout", "revcomp", "crop", "local_shift"), default="crop")
    p.add_argument("--max-length", type=int, default=450000)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--overlap-ratio", type=float, default=0.5)
    p.add_argument("--local-shift-ratio", type=float, default=0.1)
    p.add_argument("--negatives-per-anchor", type=int, default=1)
    p.add_argument("--negative-choice", choices=("first", "random"), default="first")
    p.add_argument("--max-train-examples", type=int, default=0)

    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--train-mode", choices=("full", "lora"), default="full")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", default="in_proj,out_proj,fc1,fc2")

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-trainable-fp32", dest="trainable_fp32", action="store_false")
    p.set_defaults(trainable_fp32=True)

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=0)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.chunk_size is None:
        args.chunk_size = args.max_length

    rows_by_id = read_windows_manifest(args.windows_manifest, args.split)
    min_required_len = required_parent_length(args)
    short_rows = [
        row["id"]
        for row in rows_by_id.values()
        if int(row.get("length") or (int(row["end"]) - int(row["start"]))) < min_required_len
    ]
    if short_rows:
        raise ValueError(
            f"{args.mode} mode with chunk_size={args.chunk_size:,} requires parent windows "
            f">= {min_required_len:,} bp, but {len(short_rows):,} rows are shorter. "
            f"Example: {short_rows[0]}"
        )
    examples = read_negative_manifest(args.negative_manifest, args.split, rows_by_id)
    if args.max_train_examples > 0:
        examples = examples[: args.max_train_examples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 72)
    print("HyenaDNA 1M | Step 3 explicit-negative contrastive adaptation")
    print("=" * 72)
    print(f"  Input model              : {args.model}")
    print(f"  Output                   : {args.output}")
    print(f"  Genome FASTA             : {args.genome_fasta}")
    print(f"  Windows manifest         : {args.windows_manifest}")
    print(f"  Negative manifest        : {args.negative_manifest}")
    print(f"  Split                    : {args.split}")
    print(f"  Train examples           : {len(examples):,}")
    print(f"  Mode                     : {args.mode}")
    print(f"  Max length / chunk       : {args.max_length:,} / {args.chunk_size:,} bp")
    print(f"  Negatives per anchor     : {args.negatives_per_anchor}")
    print(f"  Negative choice          : {args.negative_choice}")
    print(f"  Device / precision       : {device} / {dtype}")
    print(f"  Eval                     : disabled")

    tokenizer = load_hyena_tokenizer(args.model)
    base_model, load_path = load_hyena_causal_lm(args.model, device=device, dtype=dtype)
    print(f"  Load path                : {load_path}")
    base_model = make_hyenadna_bidirectional(base_model)
    report = inspect_hyenadna_bidirectional(base_model)
    print(f"  HyenaFilter modules      : {report.total_hyena_filters}")
    print(f"  Forward-patched modules  : {report.modules_forward_patched}")

    if hasattr(base_model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing   : enabled")

    if args.train_mode == "lora":
        print(f"  LoRA targets             : {args.lora_target_modules}")
        base_model = apply_hyena_lora(base_model, args)

    model = HyenaDNAExplicitNegativeContrastive(
        base_model=base_model,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
    ).to(device=device, dtype=dtype)

    if args.train_mode == "lora" and args.trainable_fp32:
        before_counts = promote_trainable_parameters_to_fp32(model)
        print(
            "  Trainable fp32           : promoted "
            f"(before: {', '.join(f'{k}={v/1e6:.2f}M' for k, v in sorted(before_counts.items()))})"
        )
    if args.mode == "dropout":
        n_dropout = set_dropout(model, args.dropout)
        print(f"  Dropout modules patched  : {n_dropout}")
        if n_dropout == 0:
            raise RuntimeError("Dropout mode requires at least one nn.Dropout module.")
    print(f"  Trainable parameters     : {count_trainable_parameters_m(model):.2f}M")

    dataset = LongWindowExplicitNegativeDataset(
        fasta_path=args.genome_fasta,
        rows_by_id=rows_by_id,
        examples=examples,
        negatives_per_anchor=args.negatives_per_anchor,
        negative_choice=args.negative_choice,
        seed=args.seed,
    )
    collator = ExplicitNegativeCollator(
        tokenizer=tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio,
        local_shift_ratio=args.local_shift_ratio,
    )

    os.makedirs(args.output, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    trainer = ExplicitNegativeTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\n[1/3] Training {args.mode} with explicit negatives")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"  Trainer finished         : global_step={trainer.state.global_step}", flush=True)

    print("\n[2/3] Saving checkpoint")
    config_model = _peft_base_model(model.base_model)
    config_model.config.hyena_training_stage = f"HL_step3_{args.mode}_explicitneg"
    config_model.config.hyena_explicit_negatives = True
    config_model.config.hyena_negatives_per_anchor = args.negatives_per_anchor
    if args.train_mode == "lora":
        config_model.config.hyena_train_mode = "lora"
        config_model.config.hyena_lora_target_modules = args.lora_target_modules
        config_model.config.hyena_lora_r = args.lora_r
        config_model.config.hyena_lora_alpha = args.lora_alpha
        config_model.config.hyena_lora_dropout = args.lora_dropout
    if hasattr(config_model.config, "save_pretrained"):
        config_model.config.save_pretrained(args.output)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    expected_model = os.path.join(args.output, "pytorch_model.bin")
    if not os.path.isfile(expected_model):
        raise RuntimeError(f"Expected model file was not created: {expected_model}")
    print(f"  Saved checkpoint         : {args.output}")
    print(f"  Train loss               : {train_result.training_loss:.6f}")

    print("\n[3/3] Smoke embedding check")
    model.eval()
    batch = collator([dataset[0]])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        z = model.encode(batch["input_ids_a"], batch["attention_mask_a"])
    print(f"  Embedding shape          : {tuple(z.shape)}")
    print(f"  Embedding L2 norm        : {float(torch.linalg.norm(z[0])):.6f}")


if __name__ == "__main__":
    main()
