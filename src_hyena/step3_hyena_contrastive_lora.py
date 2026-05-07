"""
HyenaDNA Step 3: contrastive adaptation.

H3: H2 adapted with identical sequence pairs and independent dropout masks.
H4: H2 adapted with a sequence and its reverse complement as the positive pair.
H5: H2 adapted with two overlapping crops from the same genomic window.
H6: H2 adapted with a center crop and a small local shift from the same window.

All variants use symmetric InfoNCE. This is intentionally a full fine-tune
analogue of the current HyenaDNA Step 2 implementation rather than a LoRA
implementation; HyenaDNA does not share DNAGPT's attention-projection module
structure.
"""

from __future__ import annotations

import argparse
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

from data_utils import VALID_BASES, _iter_fasta  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_HYENA_MODEL,
    count_trainable_parameters_m,
    extract_hidden_states,
    load_hyena_causal_lm,
    load_hyena_tokenizer,
    save_hyena_checkpoint,
)
from hyena_bidirectional import inspect_hyenadna_bidirectional, make_hyenadna_bidirectional  # noqa: E402


_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1].upper()


class RawSequenceDataset(TorchDataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


def _smoke_test_sequences(n: int = 512, seq_len: int = 3072) -> list[str]:
    bases = ["A", "C", "G", "T"]
    return ["".join(random.choices(bases, k=seq_len)) for _ in range(n)]


def _load_fasta_windows(
    fasta_path: str,
    window_bp: int,
    stride_bp: int,
    filter_n: bool,
) -> list[str]:
    stride = stride_bp if stride_bp > 0 else window_bp
    windows: list[str] = []
    skipped = 0

    for contig in _iter_fasta(fasta_path, strip_n=not filter_n):
        for start in range(0, len(contig) - window_bp + 1, stride):
            seq = contig[start : start + window_bp].upper()
            if filter_n and VALID_BASES.search(seq):
                skipped += 1
                continue
            windows.append(seq)

    print(f"  FASTA windows            : {len(windows):,}")
    if filter_n:
        print(f"  Dropped non-ACGT windows : {skipped:,}")
    return windows


def _attention_mask(tokenizer, encoding) -> torch.Tensor:
    if "attention_mask" in encoding:
        return encoding["attention_mask"]

    input_ids = encoding["input_ids"]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return input_ids.ne(int(pad_token_id)).long()


def set_dropout(model: nn.Module, p: float) -> int:
    if not (0.0 <= p < 1.0):
        raise ValueError("--dropout must be in [0, 1).")

    n_modules = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p
            n_modules += 1

    seen_config_ids: set[int] = set()
    configs = (getattr(model, "config", None), getattr(getattr(model, "base_model", None), "config", None))
    for cfg in configs:
        if cfg is None:
            continue
        if id(cfg) in seen_config_ids:
            continue
        seen_config_ids.add(id(cfg))
        for attr in (
            "dropout",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "resid_pdrop",
            "embd_pdrop",
            "summary_first_dropout",
        ):
            if hasattr(cfg, attr):
                setattr(cfg, attr, p)

    return n_modules


@dataclass
class DropoutPairCollator:
    tokenizer: object
    max_length: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [feature["sequence"].upper() for feature in features]
        enc = self.tokenizer(
            seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        return {
            "input_ids_a": enc["input_ids"],
            "attention_mask_a": _attention_mask(self.tokenizer, enc),
            "input_ids_b": enc["input_ids"].clone(),
            "attention_mask_b": _attention_mask(self.tokenizer, enc).clone(),
        }


@dataclass
class CropPairCollator:
    tokenizer: object
    chunk_size: int
    overlap_ratio: float
    max_length: int

    def __post_init__(self):
        if not (0.0 < self.overlap_ratio <= 1.0):
            raise ValueError("--overlap-ratio must be in (0, 1].")
        self.max_shift = int(round(self.chunk_size * (1.0 - self.overlap_ratio)))

    def _make_pair(self, seq: str) -> tuple[str, str]:
        if len(seq) <= self.chunk_size:
            crop = seq[: self.chunk_size]
            return crop, crop

        available_shift = min(self.max_shift, len(seq) - self.chunk_size)
        if available_shift <= 0:
            crop = seq[: self.chunk_size]
            return crop, crop

        start_a = random.randint(0, available_shift)
        start_b = random.randint(0, available_shift)
        return (
            seq[start_a : start_a + self.chunk_size],
            seq[start_b : start_b + self.chunk_size],
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs_a, seqs_b = [], []
        for feature in features:
            a, b = self._make_pair(feature["sequence"])
            seqs_a.append(a)
            seqs_b.append(b)

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
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": _attention_mask(self.tokenizer, enc_a),
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": _attention_mask(self.tokenizer, enc_b),
        }


@dataclass
class RevCompPairCollator:
    tokenizer: object
    max_length: int

    def _make_pair(self, seq: str) -> tuple[str, str]:
        seq = seq.upper()
        return seq, reverse_complement(seq)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs_a, seqs_b = [], []
        for feature in features:
            a, b = self._make_pair(feature["sequence"])
            seqs_a.append(a)
            seqs_b.append(b)

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
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": _attention_mask(self.tokenizer, enc_a),
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": _attention_mask(self.tokenizer, enc_b),
        }


@dataclass
class LocalShiftPairCollator:
    tokenizer: object
    chunk_size: int
    max_shift_ratio: float
    max_length: int

    def __post_init__(self):
        if not (0.0 <= self.max_shift_ratio < 1.0):
            raise ValueError("--local-shift-ratio must be in [0, 1).")
        self.max_shift = int(round(self.chunk_size * self.max_shift_ratio))
        self.anchor = self.max_shift

    def _make_pair(self, seq: str) -> tuple[str, str]:
        seq = seq.upper()
        if len(seq) <= self.chunk_size:
            crop = seq[: self.chunk_size]
            return crop, crop

        required_len = self.chunk_size + 2 * self.max_shift
        if self.max_shift <= 0 or len(seq) < required_len:
            crop = seq[: self.chunk_size]
            return crop, crop

        delta = random.randint(-self.max_shift, self.max_shift)
        start_a = self.anchor
        start_b = self.anchor + delta
        return (
            seq[start_a : start_a + self.chunk_size],
            seq[start_b : start_b + self.chunk_size],
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs_a, seqs_b = [], []
        for feature in features:
            a, b = self._make_pair(feature["sequence"])
            seqs_a.append(a)
            seqs_b.append(b)

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
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": _attention_mask(self.tokenizer, enc_a),
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": _attention_mask(self.tokenizer, enc_b),
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


class HyenaDNAForCropContrastive(nn.Module):
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
        # Use the Hyena backbone directly when the checkpoint is loaded as a
        # CausalLM. Calling the LM wrapper with output_hidden_states=False returns
        # logits only, while output_hidden_states=True keeps every layer output in
        # memory. The backbone path returns just the final hidden state.
        encoder = self.base_model.hyena if hasattr(self.base_model, "hyena") else self.base_model
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)
        pooled = self._mean_pool(hidden, attention_mask)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)

    @staticmethod
    def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
        labels = torch.arange(z_a.size(0), device=z_a.device)
        sim = torch.mm(z_a, z_b.t()) / temperature
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        **kwargs,
    ):
        z_a = self.encode(input_ids_a, attention_mask_a)
        z_b = self.encode(input_ids_b, attention_mask_b)
        loss = self.info_nce_loss(z_a, z_b, self.temperature)
        return type("ContrastiveOutput", (), {"loss": loss, "z_a": z_a, "z_b": z_b})()

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        save_hyena_checkpoint(self.base_model, None, save_dir)
        if self.proj is not None:
            torch.save(self.proj.state_dict(), os.path.join(save_dir, "contrastive_head.pt"))


class HyenaContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = model(**inputs).loss.detach()
        return (loss, None, None)

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
    eval_strategy = "no" if args.no_eval else ("steps" if args.eval_steps > 0 else "epoch")
    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    kwargs = dict(
        output_dir=os.path.join(args.output, "trainer_state"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        save_strategy=save_strategy,
        # Checkpoints save the Hyena backbone for Step 4 compatibility rather
        # than the transient contrastive wrapper, so avoid Trainer reloading
        # them into the wrapper at the end.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available() and not args.no_pin_memory,
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to=[] if args.no_wandb else ["wandb"],
        seed=args.seed,
        run_name=args.run_name or f"hyena_step3_{args.mode}_s{args.seed}",
        bf16=bf16_ok,
        bf16_full_eval=bf16_ok,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    if "save_safetensors" in training_args_params:
        kwargs["save_safetensors"] = False
    return TrainingArguments(**kwargs)


def load_contrastive_data(args) -> tuple[RawSequenceDataset, RawSequenceDataset]:
    chunk_size = args.chunk_size or args.max_length * 2
    max_shift = int(round(chunk_size * (1.0 - args.overlap_ratio)))
    local_shift = int(round(chunk_size * args.local_shift_ratio))
    # HyenaDNA is nucleotide-level, so max_length already corresponds roughly to bp.
    # DNAGPT uses max_length * 4 for BPE-tokenized DNA; that factor is not used here.
    if args.mode == "crop":
        window_bp = chunk_size + max_shift
    elif args.mode == "local_shift":
        window_bp = chunk_size + 2 * local_shift
    else:
        window_bp = args.max_length

    if args.smoke_test:
        print("  Data source              : synthetic smoke test")
        sequences = _smoke_test_sequences(n=512, seq_len=max(window_bp, 128))
    elif args.fasta:
        print(f"  Data source              : {args.fasta}")
        sequences = _load_fasta_windows(
            fasta_path=args.fasta,
            window_bp=window_bp,
            stride_bp=args.stride,
            filter_n=args.filter_n,
        )
    elif args.sequences_file:
        print(f"  Data source              : {args.sequences_file}")
        with open(args.sequences_file, "r", encoding="utf-8") as handle:
            sequences = [
                line.strip().upper()
                for line in handle
                if line.strip()
            ]
        if args.filter_n:
            sequences = [seq for seq in sequences if not VALID_BASES.search(seq)]
        min_len = chunk_size if args.mode in {"crop", "local_shift"} else window_bp
        sequences = [seq for seq in sequences if len(seq) >= max(32, min_len)]
    else:
        raise ValueError("Provide --fasta, --sequences-file, or --smoke-test.")

    if len(sequences) < 2:
        raise ValueError("Need at least two sequences/windows for contrastive training.")

    random.shuffle(sequences)
    n_val = max(1, int(len(sequences) * args.val_fraction))
    if n_val >= len(sequences):
        n_val = 1

    return RawSequenceDataset(sequences[n_val:]), RawSequenceDataset(sequences[:n_val])


def parse_args():
    p = argparse.ArgumentParser(description="HyenaDNA Step 3: contrastive adaptation")
    p.add_argument("--model", default=DEFAULT_HYENA_MODEL, help="H2 checkpoint or HyenaDNA model id")
    p.add_argument("--output", default="./hyena_h5_crop_contrastive")
    p.add_argument("--mode", choices=("dropout", "revcomp", "crop", "local_shift"), default="crop")

    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta", default=None)
    data.add_argument("--sequences-file", default=None, help="Plain-text file with one DNA sequence per line")
    data.add_argument("--smoke-test", action="store_true")
    p.add_argument("--hidden-path-sanity-check", action="store_true",
                   help="Compare old LM-wrapper all-hidden-state extraction with "
                        "the memory-efficient Hyena-backbone final-hidden path, then exit.")

    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.01)

    p.add_argument("--chunk-size", type=int, default=None, help="Nucleotide length of each crop.")
    p.add_argument("--overlap-ratio", type=float, default=0.5)
    p.add_argument("--local-shift-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout probability used for H3.")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj-dim", type=int, default=256)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def hidden_path_sanity_check(args, model, tokenizer, device: str, dtype: torch.dtype):
    model.eval()
    seq_len = min(args.max_length, 512)
    sequences = [
        ("ACGT" * ((seq_len // 4) + 1))[:seq_len],
        ("TGCA" * ((seq_len // 4) + 1))[:seq_len],
    ]
    enc = tokenizer(
        sequences,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        attention_mask = torch.ones_like(enc["input_ids"]) if pad_token_id is None else (enc["input_ids"] != pad_token_id).long()

    wrapper = HyenaDNAForCropContrastive(
        base_model=model,
        proj_dim=0,
        temperature=args.temperature,
    ).to(device=device, dtype=dtype)

    with torch.inference_mode():
        old_out = model(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        old_hidden = extract_hidden_states(old_out)
        old_pooled = F.normalize(wrapper._mean_pool(old_hidden, attention_mask), dim=-1)

        encoder = model.hyena if hasattr(model, "hyena") else model
        new_out = encoder(
            input_ids=enc["input_ids"],
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        new_hidden = extract_hidden_states(new_out)
        new_pooled = F.normalize(wrapper._mean_pool(new_hidden, attention_mask), dim=-1)

    max_abs = (old_pooled - new_pooled).abs().max().item()
    mean_abs = (old_pooled - new_pooled).abs().mean().item()
    cos = F.cosine_similarity(old_pooled, new_pooled, dim=-1)
    print("\n[Sanity] Hidden extraction path comparison")
    print(f"  Old hidden shape         : {tuple(old_hidden.shape)}")
    print(f"  New hidden shape         : {tuple(new_hidden.shape)}")
    print(f"  Max abs diff             : {max_abs:.8e}")
    print(f"  Mean abs diff            : {mean_abs:.8e}")
    print(f"  Cosine similarity        : {[round(float(x), 8) for x in cos]}")
    if max_abs > 1e-3:
        raise RuntimeError("Hidden path sanity check failed: old/new embeddings differ unexpectedly.")
    print("  Result                   : PASSED")


def main():
    args = parse_args()
    if args.output == "./hyena_h5_crop_contrastive":
        default_outputs = {
            "dropout": "./hyena_h3_dropout_contrastive",
            "revcomp": "./hyena_h4_revcomp_contrastive",
            "crop": "./hyena_h5_crop_contrastive",
            "local_shift": "./hyena_h6_local_shift_contrastive",
        }
        args.output = default_outputs[args.mode]
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    chunk_size = args.chunk_size or args.max_length * 2
    max_shift = int(round(chunk_size * (1.0 - args.overlap_ratio)))
    local_shift = int(round(chunk_size * args.local_shift_ratio))

    print("=" * 72)
    print("HyenaDNA  |  Step 3: Contrastive Adaptation")
    print("=" * 72)
    print(f"  Input model              : {args.model}")
    print(f"  Output                   : {args.output}")
    print(f"  Device                   : {device}")
    print(f"  Precision                : {dtype}")
    objective_names = {
        "dropout": "dropout SimCSE",
        "revcomp": "reverse-complement SimCSE",
        "crop": "crop SimCSE",
        "local_shift": "local-shift SimCSE",
    }
    objective = objective_names[args.mode]
    print(f"  Objective                : {objective} + symmetric InfoNCE")
    if args.mode == "dropout":
        print(f"  Dropout                  : {args.dropout}")
        print(f"  Window                   : {args.max_length} bp")
    elif args.mode == "crop":
        print(f"  Crop size                : {chunk_size} bp")
        print(f"  Overlap ratio            : {args.overlap_ratio:.0%}")
        print(f"  Max crop shift           : {max_shift} bp")
    elif args.mode == "local_shift":
        print(f"  Crop size                : {chunk_size} bp")
        print(f"  Local shift ratio        : {args.local_shift_ratio:.0%}")
        print(f"  Max local shift          : {local_shift} bp")
        print(f"  Window                   : {chunk_size + 2 * local_shift} bp")
    else:
        print(f"  Revcomp window           : {args.max_length} bp")
    print(f"  Temperature              : {args.temperature}")
    print(f"  Projection dim           : {args.proj_dim}")
    print(f"  Max steps / epochs       : {args.max_steps} / {args.epochs}")
    print(f"  Save steps               : {args.save_steps}")
    print(f"  Eval                     : {'disabled' if args.no_eval else f'every {args.eval_steps} steps'}")

    tokenizer = load_hyena_tokenizer(args.model)
    base_model, load_path = load_hyena_causal_lm(args.model, device=device, dtype=dtype)
    print(f"  Load path                : {load_path}")

    base_model = make_hyenadna_bidirectional(base_model)
    report = inspect_hyenadna_bidirectional(base_model)
    print(f"  HyenaFilter modules      : {report.total_hyena_filters}")
    print(f"  Forward-patched modules  : {report.modules_forward_patched}")

    if args.hidden_path_sanity_check:
        hidden_path_sanity_check(args, base_model, tokenizer, device, dtype)
        return

    if hasattr(base_model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing   : enabled")

    model = HyenaDNAForCropContrastive(
        base_model=base_model,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
    ).to(device=device, dtype=dtype)
    if args.mode == "dropout":
        n_dropout = set_dropout(model, args.dropout)
        print(f"  Dropout modules patched  : {n_dropout}")
    print(f"  Trainable parameters     : {count_trainable_parameters_m(model):.2f}M")

    print("\n[1/4] Loading contrastive data")
    train_dataset, val_dataset = load_contrastive_data(args)
    print(f"  Train samples            : {len(train_dataset):,}")
    print(f"  Validation samples       : {len(val_dataset):,}")

    if args.mode == "dropout":
        collator = DropoutPairCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    elif args.mode == "crop":
        collator = CropPairCollator(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap_ratio=args.overlap_ratio,
            max_length=args.max_length,
        )
    elif args.mode == "local_shift":
        collator = LocalShiftPairCollator(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_shift_ratio=args.local_shift_ratio,
            max_length=args.max_length,
        )
    else:
        collator = RevCompPairCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

    os.makedirs(args.output, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    training_args = build_training_args(args)
    trainer = HyenaContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\n[2/4] Training {args.mode} SimCSE adaptation")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"  Trainer finished         : global_step={trainer.state.global_step}", flush=True)

    stages = {
        "dropout": "H3_dropout_contrastive",
        "revcomp": "H4_revcomp_contrastive",
        "crop": "H5_crop_contrastive",
        "local_shift": "H6_local_shift_contrastive",
    }
    stage = stages[args.mode]
    print(f"\n[3/4] Saving {stage} checkpoint")
    model.base_model.config.hyena_training_stage = stage
    if hasattr(model.base_model.config, "save_pretrained"):
        model.base_model.config.save_pretrained(args.output)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    expected_model = os.path.join(args.output, "pytorch_model.bin")
    if not os.path.isfile(expected_model):
        raise RuntimeError(f"Expected model file was not created: {expected_model}")
    print(f"  Saved checkpoint         : {args.output}", flush=True)
    print(f"  Model file               : {expected_model}", flush=True)
    print(f"  Train loss               : {train_result.training_loss:.6f}", flush=True)

    print("\n[4/4] Smoke embedding check")
    model.eval()
    batch = collator([{"sequence": "ACGT" * max(64, chunk_size // 4)}])
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        z = model.encode(batch["input_ids_a"], batch["attention_mask_a"])
    print(f"  Embedding shape          : {tuple(z.shape)}")
    print(f"  Embedding L2 norm        : {float(torch.linalg.norm(z[0])):.6f}")


if __name__ == "__main__":
    main()
