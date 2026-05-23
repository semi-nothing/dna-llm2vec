"""
Evo Step 3: contrastive adaptation with LoRA for E3--E6.

Modes:
  dropout    -> E3, identical sequence pairs with independent dropout masks
  revcomp    -> E4, sequence and reverse-complement pair
  crop       -> E5, overlapping long-context crops
  local_shift-> E6, nearby shifted crops
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    DEFAULT_EVO_MODEL,
    VALID_BASES_RE,
    count_trainable_parameters_m,
    evo_hidden_states,
    infer_hidden_dim,
    infer_lora_target_modules,
    load_evo_causal_lm,
    load_evo_tokenizer,
    mean_pool_embeddings,
    save_evo_checkpoint,
)

RC_TABLE = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]


class RawSequenceDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


def load_fasta_windows(path: str, window_bp: int, stride_bp: int, filter_n: bool) -> list[str]:
    seqs: list[str] = []
    current: list[str] = []
    stride = stride_bp or window_bp

    def flush():
        if not current:
            return
        seq = "".join(current).upper()
        for start in range(0, max(0, len(seq) - window_bp + 1), stride):
            window = seq[start : start + window_bp]
            if len(window) == window_bp and (not filter_n or not VALID_BASES_RE.search(window)):
                seqs.append(window)
        current.clear()

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
            else:
                current.append(line)
    flush()
    return seqs


@dataclass
class PairCollator:
    tokenizer: object
    mode: str
    max_length: int
    chunk_size: int
    overlap_ratio: float
    local_shift_ratio: float

    def _tokenize(self, seqs: list[str]):
        return self.tokenizer(
            seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [item["sequence"].upper() for item in features]
        if self.mode == "dropout":
            a, b = seqs, seqs
        elif self.mode == "revcomp":
            a, b = seqs, [reverse_complement(s) for s in seqs]
        elif self.mode == "crop":
            a, b = [], []
            max_shift = int(round(self.chunk_size * (1.0 - self.overlap_ratio)))
            for seq in seqs:
                if len(seq) <= self.chunk_size:
                    a.append(seq[: self.chunk_size])
                    b.append(seq[: self.chunk_size])
                    continue
                available_shift = min(max_shift, len(seq) - self.chunk_size)
                if available_shift <= 0:
                    a.append(seq[: self.chunk_size])
                    b.append(seq[: self.chunk_size])
                    continue
                start_a = random.randint(0, available_shift)
                start_b = random.randint(0, available_shift)
                a.append(seq[start_a : start_a + self.chunk_size])
                b.append(seq[start_b : start_b + self.chunk_size])
        elif self.mode == "local_shift":
            a, b = [], []
            shift_max = int(round(self.chunk_size * self.local_shift_ratio))
            anchor = shift_max
            for seq in seqs:
                required_len = self.chunk_size + 2 * shift_max
                if shift_max <= 0 or len(seq) < required_len:
                    a.append(seq[: self.chunk_size])
                    b.append(seq[: self.chunk_size])
                    continue
                delta = random.randint(-shift_max, shift_max)
                start_b = anchor + delta
                a.append(seq[anchor : anchor + self.chunk_size])
                b.append(seq[start_b : start_b + self.chunk_size])
        else:
            raise ValueError(self.mode)
        enc_a = self._tokenize(a)
        enc_b = self._tokenize(b)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a.get("attention_mask"),
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b.get("attention_mask"),
        }


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_dropout(model: nn.Module, p: float) -> int:
    n = 0
    for name, module in model.named_modules():
        if "lora_" in name:
            continue
        if isinstance(module, nn.Dropout):
            module.p = p
            n += 1
    return n


def build_lora_config(model, args):
    from peft import LoraConfig

    targets = infer_lora_target_modules(model, args.lora_target_modules)
    print(f"  LoRA targets             : {targets}")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias="none",
    )


class EvoForContrastiveLoRA(nn.Module):
    def __init__(self, base_model, lora_config, temperature: float, proj_dim: int):
        super().__init__()
        from peft import get_peft_model

        self.peft_model = get_peft_model(base_model, lora_config)
        self.config = base_model.config
        hidden_dim = infer_hidden_dim(base_model)
        self.proj = ProjectionHead(hidden_dim, proj_dim) if proj_dim > 0 else None
        self.temperature = temperature

    def encode(self, input_ids, attention_mask=None):
        hidden = evo_hidden_states(self.peft_model, input_ids, attention_mask)
        pooled = mean_pool_embeddings(hidden, attention_mask)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)

    def forward(self, input_ids_a, attention_mask_a=None, input_ids_b=None, attention_mask_b=None, **kwargs):
        z_a = self.encode(input_ids_a, attention_mask_a)
        z_b = self.encode(input_ids_b, attention_mask_b)
        logits = z_a @ z_b.T / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return type("ContrastiveOutput", (), {"loss": loss})()

    def save_pretrained(self, output_dir: str, merge: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        if merge:
            model = self.peft_model.merge_and_unload()
            save_evo_checkpoint(model, None, output_dir)
        else:
            self.peft_model.save_pretrained(output_dir)
            if self.proj is not None:
                torch.save(self.proj.state_dict(), os.path.join(output_dir, "proj_head.pt"))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.peft_model, "enable_input_require_grads"):
            self.peft_model.enable_input_require_grads()
        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs or {"use_reentrant": False})


class EvoContrastiveTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir, merge=False)
        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if proc is not None:
            proc.save_pretrained(output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evo Step 3: contrastive LoRA")
    p.add_argument("--model", default=DEFAULT_EVO_MODEL)
    p.add_argument("--output", default="./evo_e5_crop_contrastive_lora")
    p.add_argument("--mode", choices=("dropout", "revcomp", "crop", "local_shift"), default="crop")
    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta")
    data.add_argument("--sequences-file")
    data.add_argument("--smoke-test", action="store_true")
    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--stride", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--overlap-ratio", type=float, default=0.5)
    p.add_argument("--local-shift-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load_data(args):
    chunk = args.chunk_size or args.max_length
    if args.mode == "crop":
        window = chunk + int(round(chunk * (1.0 - args.overlap_ratio)))
    elif args.mode == "local_shift":
        window = chunk + 2 * int(round(chunk * args.local_shift_ratio))
    else:
        window = args.max_length
    if args.smoke_test:
        seqs = ["".join(random.choices("ACGT", k=window)) for _ in range(256)]
    elif args.fasta:
        seqs = load_fasta_windows(args.fasta, window, args.stride, args.filter_n)
    elif args.sequences_file:
        with open(args.sequences_file, "r", encoding="utf-8") as handle:
            seqs = [line.strip().upper() for line in handle if line.strip()]
        if args.filter_n:
            seqs = [s for s in seqs if not VALID_BASES_RE.search(s)]
    else:
        raise ValueError("Provide --fasta, --sequences-file or --smoke-test.")
    if len(seqs) < 2:
        raise ValueError("Need at least two sequences.")
    random.shuffle(seqs)
    n_val = max(1, int(len(seqs) * 0.01))
    return RawSequenceDataset(seqs[n_val:]), RawSequenceDataset(seqs[:n_val])


def main():
    args = parse_args()
    if args.output == "./evo_e5_crop_contrastive_lora":
        args.output = {
            "dropout": "./evo_e3_dropout_contrastive_lora",
            "revcomp": "./evo_e4_revcomp_contrastive_lora",
            "crop": "./evo_e5_crop_contrastive_lora",
            "local_shift": "./evo_e6_local_shift_contrastive_lora",
        }[args.mode]
    set_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = load_evo_tokenizer(args.model)
    base_model, _ = load_evo_causal_lm(args.model, device=device, dtype=dtype)
    wrapper = EvoForContrastiveLoRA(base_model, build_lora_config(base_model, args), args.temperature, args.proj_dim)
    wrapper = wrapper.to(device=device)
    if args.mode == "dropout":
        print(f"  Dropout modules patched  : {set_dropout(wrapper, args.dropout)}")
    print(f"  Trainable parameters     : {count_trainable_parameters_m(wrapper):.2f}M")
    train_ds, val_ds = load_data(args)
    collator = PairCollator(
        tokenizer=tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        chunk_size=args.chunk_size or args.max_length,
        overlap_ratio=args.overlap_ratio,
        local_shift_ratio=args.local_shift_ratio,
    )
    eval_strategy = "no" if args.no_eval else "steps"
    kwargs = dict(
        output_dir=os.path.join(args.output, "trainer_state"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=(dtype is torch.bfloat16),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available() and not args.no_pin_memory,
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.run_name or f"evo_step3_{args.mode}_lora_s{args.seed}",
        seed=args.seed,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    training_args = TrainingArguments(**kwargs)
    trainer_kwargs = dict(
        model=wrapper,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if args.no_eval else val_ds,
        data_collator=collator,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = EvoContrastiveTrainer(**trainer_kwargs)
    trainer.train()
    stage_num = {"dropout": 3, "revcomp": 4, "crop": 5, "local_shift": 6}[args.mode]
    wrapper.peft_model.base_model.model.config.evo_training_stage = f"E{stage_num}_{args.mode}_lora"
    wrapper.save_pretrained(args.output, merge=True)
    tokenizer.save_pretrained(args.output)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved merged Evo Step 3 checkpoint to {args.output}")


if __name__ == "__main__":
    main()
