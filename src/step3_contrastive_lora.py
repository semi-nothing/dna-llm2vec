"""
DNA-LLM2Vec  |  Step 3 (LoRA): Contrastive Fine-tuning LLM2Vec style
========================================================================
LoRA-based contrastive training with four improvements over step3_contrastive.py:

  1. LoRA (PEFT): only adapter weights are updated during contrastive training,
     consistent with step2_mntp_lora.py and LLM2Vec's training recipe.

  2. Dropout=0.3 for dropout mode (M3): LLM2Vec found 0.1 insufficient for
     dropout-based SimCSE; 0.3 creates diverse enough positive pairs.
     (revcomp mode is unaffected positive pairs are different sequences.)

  3. device_map removed: loading with device_map="auto" wraps the model in
     an Accelerate dispatch object, causing device mismatches when extracting
     the transformer submodule. Load on CPU, move to device manually.

  4. --max-steps: train for a fixed number of steps (LLM2Vec uses 1000)
     instead of full epochs. Defaults to -1 (use --epochs instead).

Checkpoint layout:
  <output>/checkpoint-N/
    adapter_model.safetensors   LoRA adapter weights
    adapter_config.json
    proj_head.pt                projection head state dict (if used)
  <output>/
    config.json                 final merged GPT2LMHeadModel (encoder only)
    model.safetensors           (projection head NOT saved, as per original design)

Usage:
  # M3 dropout SimCSE (LoRA, dropout=0.3):
  uv run python src/step3_contrastive_lora.py \\
      --model  ./mntp_dnagpt_lora \\
      --fasta  ./data/hg38.fa \\
      --output ./contrastive_dnagpt_dropout_lora \\
      --mode   dropout --max-steps 1000

  # M4 reverse complement (LoRA, proposed method):
  uv run python src/step3_contrastive_lora.py \\
      --model  ./mntp_dnagpt_lora \\
      --fasta  ./data/hg38.fa \\
      --output ./contrastive_dnagpt_revcomp_lora \\
      --mode   revcomp --max-steps 1000

  # Smoke-test:
  uv run python src/step3_contrastive_lora.py \\
      --model ./mntp_dnagpt_lora --output ./test_out \\
      --mode dropout --smoke-test
"""

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(__file__))
from step1_bidirectional import patch_to_bidirectional
from data_utils import _iter_fasta, _chunk_sequence


# ── Reverse complement ────────────────────────────────────────────────────────

_RC_TABLE = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


# ── Dropout override ──────────────────────────────────────────────────────────

def set_dropout(model: nn.Module, p: float, skip_lora: bool = True):
    """
    Override nn.Dropout layers in a model with probability p.

    Used to set dropout=0.3 for SimCSE dropout mode (M3), matching
    LLM2Vec's finding that 0.1 (the DNAGPT default) is insufficient
    to create diverse enough positive pairs for contrastive learning.

    skip_lora=True (default): skips LoRA adapter dropout layers so their
    carefully tuned lora_dropout value is not overridden.

    Note: only affects dropout during model.train() eval() disables all dropout.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            if skip_lora and "lora_" in name:
                continue
            module.p = p
            count += 1
    print(f"  set_dropout: updated {count} Dropout layers to p={p}")


# ── LoRA config ───────────────────────────────────────────────────────────────

def build_lora_config(args):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        fan_in_fan_out=True,                   # required for GPT-2 Conv1D layers
    )


# ── Datasets (unchanged from step3_contrastive.py) ───────────────────────────

class RawSequenceDataset(TorchDataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


def _smoke_test_sequences(n: int = 2000, min_len: int = 64, max_len: int = 512) -> list[str]:
    bases = "ACGT"
    rng   = random.Random(42)
    return [
        "".join(rng.choices(bases, k=rng.randint(min_len, max_len)))
        for _ in range(n)
    ]


def _extract_sequences_from_fasta(
    fasta_path: str,
    max_length_bp: int,
    stride_bp: int,
    max_samples: Optional[int] = None,
    filter_n: bool = False,
) -> list[str]:
    """
    Stream FASTA, chunk into fixed-length windows, return list of raw DNA strings.

    Args:
        filter_n : if True, discard chunks containing any N or ambiguous base
                   (DNABERT-2 style). When False, non-ACGT chars are stripped
                   before chunking (legacy; creates artificial junctions).
    """
    from data_utils import VALID_BASES as _VALID_BASES

    sequences = []
    effective_stride = stride_bp if stride_bp > 0 else max_length_bp
    n_filtered = 0

    for contig in _iter_fasta(fasta_path, strip_n=not filter_n):
        for chunk in _chunk_sequence(contig, max_length_bp, effective_stride):
            if filter_n and _VALID_BASES.search(chunk):
                n_filtered += 1
                continue
            sequences.append(chunk)
            if max_samples and len(sequences) >= max_samples:
                if filter_n and n_filtered:
                    print(f"  [filter_n] dropped {n_filtered:,} N-containing chunks")
                return sequences

    if filter_n and n_filtered:
        print(f"  [filter_n] dropped {n_filtered:,} N-containing chunks")
    return sequences



# ── Collators (unchanged) ─────────────────────────────────────────────────────

@dataclass
class DropoutPairCollator:
    tokenizer: object
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["sequence"] for f in features]
        enc  = self.tokenizer(
            seqs, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        return {
            "input_ids_a":      enc["input_ids"],
            "attention_mask_a": enc["attention_mask"],
            "input_ids_b":      enc["input_ids"].clone(),
            "attention_mask_b": enc["attention_mask"].clone(),
        }


@dataclass
class RevCompPairCollator:
    tokenizer: object
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        seqs  = [f["sequence"] for f in features]
        rcs   = [reverse_complement(s) for s in seqs]
        enc_a = self.tokenizer(
            seqs, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        enc_b = self.tokenizer(
            rcs, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }


@dataclass
class CropPairCollator:
    """Positive pairs = two overlapping crops from the same genomic window.

    Each stored sequence is `long_size = chunk_size + max_shift` bp.
    Two crop start positions are sampled independently from [0, max_shift],
    guaranteeing overlap chunk_size * overlap_ratio.
    """
    tokenizer:     object
    chunk_size:    int          # bp length of each crop
    overlap_ratio: float = 0.5
    max_length:    int   = 512

    def __post_init__(self):
        self.max_shift = int(self.chunk_size * (1.0 - self.overlap_ratio))

    def __call__(self, features: list[dict]) -> dict:
        seqs_a, seqs_b = [], []
        for f in features:
            seq = f["sequence"]
            if len(seq) < self.chunk_size:
                # sequence too short use as-is for both sides
                seqs_a.append(seq)
                seqs_b.append(seq)
            elif self.max_shift == 0 or len(seq) < self.chunk_size + self.max_shift:
                # no room to shift identical crops (valid degenerate positive)
                seqs_a.append(seq[:self.chunk_size])
                seqs_b.append(seq[:self.chunk_size])
            else:
                a = random.randint(0, self.max_shift)
                b = random.randint(0, self.max_shift)
                seqs_a.append(seq[a : a + self.chunk_size])
                seqs_b.append(seq[b : b + self.chunk_size])
        enc_a = self.tokenizer(
            seqs_a, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        enc_b = self.tokenizer(
            seqs_b, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }


@dataclass
class LocalShiftPairCollator:
    """Positive pairs = a center crop plus a small local shift from the same window."""

    tokenizer: object
    chunk_size: int
    max_shift_ratio: float = 0.1
    max_length: int = 512

    def __post_init__(self):
        if not (0.0 <= self.max_shift_ratio < 1.0):
            raise ValueError("max_shift_ratio must be in [0, 1)")
        self.max_shift = int(self.chunk_size * self.max_shift_ratio)
        self.anchor = self.max_shift

    def __call__(self, features: list[dict]) -> dict:
        seqs_a, seqs_b = [], []
        for f in features:
            seq = f["sequence"]
            if len(seq) < self.chunk_size:
                seqs_a.append(seq)
                seqs_b.append(seq)
                continue
            if self.max_shift == 0 or len(seq) < self.chunk_size + 2 * self.max_shift:
                base = seq[: self.chunk_size]
                seqs_a.append(base)
                seqs_b.append(base)
                continue

            delta = random.randint(-self.max_shift, self.max_shift)
            start_a = self.anchor
            start_b = self.anchor + delta
            seqs_a.append(seq[start_a : start_a + self.chunk_size])
            seqs_b.append(seq[start_b : start_b + self.chunk_size])

        enc_a = self.tokenizer(
            seqs_a, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        enc_b = self.tokenizer(
            seqs_b, truncation=True, max_length=self.max_length,
            padding="longest", return_tensors="pt",
        )
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }


# ── Pooler / Projection head (unchanged) ─────────────────────────────────────

class MeanPooler(nn.Module):
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask       = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        sum_hidden = (hidden * mask).sum(dim=1)
        lengths    = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / lengths


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Contrastive model (LoRA version) ─────────────────────────────────────────

class DNAGPTForContrastiveLora(nn.Module):
    """
    LoRA-adapted bidirectional DNAGPT for contrastive training.

    Architecture:
      PeftModel (TaskType.FEATURE_EXTRACTION)
        └── GPT2LMHeadModel (frozen base + bidirectional patch)
              └── transformer.h[*].attn.c_attn / c_proj  LoRA adapters

    encode() routes through peft_model.base_model.model.transformer,
    which includes the LoRA-modified attention layers.

    save_pretrained():
      Merges LoRA into base weights and saves the GPT-2 encoder only
      (projection head is discarded), matching step3_contrastive.py behaviour.
    """

    _keys_to_ignore_on_save = None

    def __init__(
        self,
        base_model,
        lora_config,
        proj_dim: int = 256,
        temperature: float = 0.05,
        mode: str = "dropout",
    ):
        super().__init__()
        from peft import get_peft_model
        self.peft_model  = get_peft_model(base_model, lora_config)
        self.config      = base_model.config
        self.pooler      = MeanPooler()
        self.temperature = temperature
        self.mode        = mode

        hidden_dim = self.config.hidden_size   # 768 for human_gpt2-v1
        if proj_dim > 0:
            self.proj      = ProjectionHead(hidden_dim, hidden_dim, proj_dim)
            self.embed_dim = proj_dim
        else:
            self.proj      = None
            self.embed_dim = hidden_dim

    def _transformer(self):
        """Return the LoRA-adapted GPT2Model (bidirectional attention preserved)."""
        return self.peft_model.base_model.model.transformer

    # ── Gradient checkpointing ────────────────────────────────────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.peft_model.enable_input_require_grads()
        kwargs = {"use_reentrant": False}
        if gradient_checkpointing_kwargs:
            kwargs.update(gradient_checkpointing_kwargs)
        self._transformer().gradient_checkpointing_enable(kwargs)

    def gradient_checkpointing_disable(self):
        self._transformer().gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self._transformer(), "gradient_checkpointing", False)

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised mean-pooled embeddings of shape (B, embed_dim)."""
        out    = self._transformer()(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                    # (B, T, D)
        pooled = self.pooler(hidden, attention_mask)      # (B, D)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)

    # ── InfoNCE loss ──────────────────────────────────────────────────────────

    @staticmethod
    def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
        B      = z_a.size(0)
        sim    = torch.mm(z_a, z_b.t()) / temperature
        labels = torch.arange(B, device=z_a.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        **kwargs,
    ):
        z_a  = self.encode(input_ids_a, attention_mask_a)
        z_b  = self.encode(input_ids_b, attention_mask_b)
        loss = self.info_nce_loss(z_a, z_b, self.temperature)
        return type("ContrastiveOutput", (), {"loss": loss, "z_a": z_a, "z_b": z_b})()

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str, merge: bool = True):
        """
        merge=True  (final): merge LoRA save encoder only as GPT2LMHeadModel.
                             Projection head is NOT saved (matches original design).
        merge=False (ckpt) : save LoRA adapter weights + projection head separately.
        """
        os.makedirs(save_dir, exist_ok=True)
        if merge:
            merged = self.peft_model.merge_and_unload()   # GPT2LMHeadModel
            merged.config.update(self.config.to_dict())
            merged.save_pretrained(save_dir)
        else:
            self.peft_model.save_pretrained(save_dir)
            self.peft_model.base_model.model.config.save_pretrained(save_dir)
            if self.proj is not None:
                torch.save(
                    self.proj.state_dict(),
                    os.path.join(save_dir, "proj_head.pt"),
                )

    def print_trainable_parameters(self):
        self.peft_model.print_trainable_parameters()


# ── Custom Trainer ────────────────────────────────────────────────────────────

class ContrastiveTrainerLora(Trainer):
    """
    Trainer for DNAGPTForContrastiveLora.

    Checkpoint saves store LoRA adapter weights + projection head.
    _load_best_model hot-swaps adapter weights without reloading the base model.
    """

    def __init__(self, proj_dim: int, temperature: float, mode: str, **kwargs):
        super().__init__(**kwargs)
        self._proj_dim    = proj_dim
        self._temperature = temperature
        self._mode        = mode

    def _save(self, output_dir: str, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, merge=False)   # LoRA adapters + proj head
        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if proc is not None:
            proc.save_pretrained(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir: str):
        """
        Intentionally skip optimizer / scheduler checkpointing.

        For this LoRA contrastive stage we mainly need adapter/model checkpoints.
        On some filesystems, saving optimizer.pt has caused intermittent
        torch.save zip-writer failures that abort otherwise healthy runs.
        """
        return

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        ckpt = self.state.best_model_checkpoint

        # Reload LoRA adapter weights (no base model reload needed)
        try:
            from peft import set_peft_model_state_dict, load_peft_weights
            adapter_weights = load_peft_weights(ckpt)
            set_peft_model_state_dict(self.model.peft_model, adapter_weights)
            print(f"  Loaded best LoRA adapter from: {ckpt}")
        except Exception as e:
            print(f"  [warn] LoRA reload failed ({e}), skipping best model reload")

        # Reload projection head if saved
        proj_path = os.path.join(ckpt, "proj_head.pt")
        if os.path.exists(proj_path) and self.model.proj is not None:
            device = next(self.model.proj.parameters()).device
            self.model.proj.load_state_dict(
                torch.load(proj_path, map_location=device, weights_only=True)
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss    = outputs.loss.detach()
        return (loss, None, None)


# ── Evaluation metrics callback (unchanged logic) ────────────────────────────

class ContrastiveMetricsCallback(TrainerCallback):
    def __init__(self, val_dataset, collator, model_ref, args, n_samples: int = 256):
        self.val_dataset     = val_dataset
        self.collator        = collator
        self.model_ref       = model_ref
        self.train_args      = args
        self.n_samples       = n_samples
        self._step_start     = None
        self._step_start_idx = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self._step_start is None:
            self._step_start     = time.time()
            self._step_start_idx = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        extra = {}

        if "loss" in logs and self._step_start is not None:
            elapsed    = time.time() - self._step_start
            steps_done = max(state.global_step - self._step_start_idx, 1)
            if elapsed > 0:
                extra["train_steps_per_sec"] = round(steps_done / elapsed, 3)
            self._step_start     = time.time()
            self._step_start_idx = state.global_step

        if "eval_loss" in logs:
            try:
                extra["eval_perplexity"] = round(math.exp(min(logs["eval_loss"], 20)), 4)
            except OverflowError:
                extra["eval_perplexity"] = float("inf")
            try:
                extra.update(self._compute_embedding_metrics())
            except Exception as e:
                print(f"  [metrics callback] {e}")

        if extra and args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(extra, step=state.global_step)
            except Exception:
                pass
        for k, v in extra.items():
            logs[k] = v

    @torch.no_grad()
    def _compute_embedding_metrics(self) -> dict:
        model = self.model_ref
        model.eval()
        n       = min(self.n_samples, len(self.val_dataset))
        batch   = self.collator([self.val_dataset[i] for i in range(n)])
        device  = next(model.parameters()).device
        batch   = {k: v.to(device) for k, v in batch.items()}
        z_a     = model.encode(batch["input_ids_a"], batch["attention_mask_a"])
        z_b     = model.encode(batch["input_ids_b"], batch["attention_mask_b"])

        alignment  = (z_a - z_b).pow(2).sum(dim=-1).mean().item()
        z_all      = torch.cat([z_a, z_b], dim=0)
        sq_pdist   = torch.pdist(z_all, p=2).pow(2)
        uniformity = torch.log(torch.exp(-2.0 * sq_pdist).mean() + 1e-9).item()
        sim_matrix = torch.mm(z_a, z_b.t())
        B          = z_a.size(0)
        off_diag   = sim_matrix.masked_fill(torch.eye(B, device=device).bool(), 0.0)
        avg_cos    = off_diag.sum() / (B * (B - 1)) if B > 1 else torch.tensor(0.0)

        return {
            "eval_alignment":      round(alignment, 6),
            "eval_uniformity":     round(uniformity, 6),
            "eval_avg_cosine_sim": round(avg_cos.item(), 6),
        }


# ── Training arguments ────────────────────────────────────────────────────────

def build_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output,
        # max_steps takes priority over num_train_epochs when > 0
        max_steps=args.max_steps,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=args.grad_ckpt,
        torch_compile=args.compile,
        dataloader_num_workers=4,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output, "logs"),
        logging_steps=args.log_steps,
        report_to=["wandb"] if not args.no_wandb else [],
        run_name=args.run_name or (
            f"step3_lora_{args.mode}_s{args.seed}"
            + (f"_r{args.repeat_index}" if args.repeat_index is not None else "")
        ),
        seed=args.seed,
        prediction_loss_only=True,
        remove_unused_columns=False,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_contrastive_data(args, tokenizer) -> tuple:
    if args.smoke_test:
        if args.mode == "local_shift":
            print("[Data] Smoke-test synthetic local_shift sequences")
            chunk_size = args.chunk_size or args.max_length * 2
            max_shift = int(chunk_size * args.local_shift_ratio)
            seqs = _smoke_test_sequences(
                n=2000,
                min_len=chunk_size + 2 * max_shift,
                max_len=max((chunk_size + 2 * max_shift) * 2, chunk_size + 2 * max_shift),
            )
            split = int(len(seqs) * 0.99)
            return RawSequenceDataset(seqs[:split]), RawSequenceDataset(seqs[split:])
        print("[Data] Smoke-test synthetic sequences")
        seqs  = _smoke_test_sequences(n=2000)
        split = int(len(seqs) * 0.99)
        return RawSequenceDataset(seqs[:split]), RawSequenceDataset(seqs[split:])

    if args.fasta:
        print(f"[Data] FASTA: {args.fasta}")
        if args.mode == "crop":
            chunk_size = args.chunk_size or args.max_length * 2
            max_shift  = int(chunk_size * (1.0 - args.overlap_ratio))
            chunk_bp   = chunk_size + max_shift          # long window per sample
            print(f"  crop mode: chunk_size={chunk_size} bp, "
                  f"overlap_ratio={args.overlap_ratio:.0%}, "
                  f"long_window={chunk_bp} bp")
        elif args.mode == "local_shift":
            chunk_size = args.chunk_size or args.max_length * 2
            max_shift  = int(chunk_size * args.local_shift_ratio)
            chunk_bp   = chunk_size + 2 * max_shift
            print(f"  local_shift mode: chunk_size={chunk_size} bp, "
                  f"max_shift_ratio={args.local_shift_ratio:.0%}, "
                  f"max_shift={max_shift} bp, long_window={chunk_bp} bp")
        else:
            chunk_bp   = args.max_length * 4
        stride_bp = args.stride if args.stride > 0 else chunk_bp
        seqs      = _extract_sequences_from_fasta(
            args.fasta, max_length_bp=chunk_bp, stride_bp=stride_bp,
            filter_n=args.filter_n,
        )
        print(f"  {len(seqs):,} chunks loaded")
        random.seed(args.seed)
        random.shuffle(seqs)
        n_val = max(1, int(len(seqs) * args.val_fraction))
        return RawSequenceDataset(seqs[n_val:]), RawSequenceDataset(seqs[:n_val])

    if args.dataset:
        if args.mode == "local_shift":
            raise ValueError("--mode local_shift currently requires --fasta or --smoke-test")
        print(f"[Data] HuggingFace: {args.dataset}")
        from datasets import load_dataset
        raw  = load_dataset(args.dataset)
        col  = args.sequence_column

        def get_seqs(split):
            return [ex[col] for ex in raw[split] if len(ex[col]) >= 32]

        train_seqs = get_seqs("train")
        if "validation" in raw:
            val_seqs = get_seqs("validation")
        else:
            n_val      = max(1, int(len(train_seqs) * args.val_fraction))
            val_seqs   = train_seqs[:n_val]
            train_seqs = train_seqs[n_val:]
        return RawSequenceDataset(train_seqs), RawSequenceDataset(val_seqs)

    raise ValueError("Provide --fasta, --dataset, or --smoke-test")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    return max(ckpts, key=lambda p: int(p.split("-")[-1])) if ckpts else None


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DNA-LLM2Vec Step 3 (LoRA): Contrastive")

    # Model / data
    p.add_argument("--model",            required=True)
    p.add_argument("--output",           required=True)
    p.add_argument("--attention-mode", choices=["bidir", "causal"], default="bidir",
                   help="Attention mask used during contrastive training. "
                        "Default 'bidir' preserves the main M3-M6 pipeline. "
                        "Use 'causal' for M0+contrastive ablations.")
    p.add_argument("--fasta",            default=None)
    p.add_argument("--dataset",          default=None)
    p.add_argument("--sequence-column",  default="sequence")
    p.add_argument("--smoke-test",       action="store_true")

    # Contrastive
    p.add_argument("--mode",        default="revcomp",
                   choices=["dropout", "revcomp", "crop", "local_shift"])
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj-dim",    type=int,   default=256)

    # Dropout only active in dropout mode
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Attention/embedding dropout for dropout mode. "
                        "LLM2Vec recommends 0.3 (default). Ignored in revcomp/crop mode.")
    p.add_argument("--view-dropout", type=float, default=None,
                   help="Optional dropout override for revcomp/crop/local_shift. "
                        "Use this to combine local_shift positives with dropout view noise.")

    # Crop only active in crop mode
    p.add_argument("--overlap-ratio", type=float, default=0.5,
                   help="[crop mode] Minimum overlap between the two crops as a fraction "
                        "of --chunk-size (0 < ratio < 1). Default: 0.5 (50%% overlap).")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="[crop mode] Nucleotide length of each crop. "
                        "Default: max_length * 2 bp (half the loaded window).")

    # Local-shift only active in local_shift mode
    p.add_argument("--local-shift-ratio", type=float, default=0.1,
                   help="[local_shift mode] Maximum absolute shift as a fraction of --chunk-size. "
                        "Default: 0.1 (10%% of the crop length).")

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Data
    p.add_argument("--max-length",   type=int,   default=1024)
    p.add_argument("--stride",       type=int,   default=0)
    p.add_argument("--filter-n",     action="store_true", default=False,
                   help="Drop FASTA chunks containing any N or ambiguous base "
                        "(DNABERT-2 style). Without this flag, non-ACGT characters "
                        "are stripped in-place, creating artificial junctions.")
    p.add_argument("--val-fraction", type=float, default=0.01)

    # Training
    p.add_argument("--max-steps",  type=int,   default=1000,
                   help="Train for this many steps (LLM2Vec default=1000). "
                        "-1 = use --epochs instead.")
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--batch-size", type=int,   default=128,
                   help="Per-device batch size (LLM2Vec uses 128, default=128)")
    p.add_argument("--grad-accum", type=int,   default=1)
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Learning rate (LoRA adapter updates, default=1e-4)")
    p.add_argument("--eval-steps", type=int,   default=100)
    p.add_argument("--save-steps", type=int,   default=100)
    p.add_argument("--log-steps",  type=int,   default=10)
    p.add_argument("--compile",    action="store_true", default=False)
    p.add_argument("--grad-ckpt",  action="store_true", default=False)
    p.add_argument("--patience",   type=int,   default=0)
    p.add_argument("--resume",     default=None)
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--run-name",   default=None,
                   help="Optional W&B run name. If omitted, a descriptive name is generated.")
    p.add_argument("--repeat-index", type=int, default=None,
                   help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    p.add_argument("--no-wandb",   action="store_true")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 62)
    print(f"DNA-LLM2Vec  |  Step 3 (LoRA) Contrastive ({args.mode})")
    print("=" * 62)
    if device == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mode : {args.mode}  |  τ={args.temperature}  |  proj_dim={args.proj_dim}")
    print(f"  Attention : {args.attention_mode}")
    print(f"  LoRA : r={args.lora_r}, alpha={args.lora_alpha}")
    if args.mode == "dropout":
        print(f"  Dropout : {args.dropout}  (overrides DNAGPT default 0.1)")
    if args.mode == "crop":
        _cs = args.chunk_size or args.max_length * 2
        print(f"  Crop : overlap_ratio={args.overlap_ratio:.0%}, chunk_size={_cs} bp, "
              f"min_overlap={int(_cs * args.overlap_ratio)} bp")
    if args.mode == "local_shift":
        _cs = args.chunk_size or args.max_length * 2
        print(f"  Local shift : chunk_size={_cs} bp, "
              f"max_shift_ratio={args.local_shift_ratio:.0%}, max_shift={int(_cs * args.local_shift_ratio)} bp")
    if args.view_dropout is not None and args.mode != "dropout":
        print(f"  View dropout : {args.view_dropout}  (applied with {args.mode} positives)")
    steps_info = f"{args.max_steps} steps" if args.max_steps > 0 else f"{args.epochs} epochs"
    print(f"  Training : {steps_info}, batch={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}")
    print(f"  Max length : {args.max_length} tokens")
    print("=" * 62)

    # ── 1. Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load base model + apply LoRA ──────────────────────────────────────
    print(f"\n[2/5] Loading base model + applying LoRA")

    # Load on CPU, move to device manually avoids device_map dispatch issues
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    if args.attention_mode == "bidir":
        base = patch_to_bidirectional(base)
    else:
        base.config.is_causal = True
        base.config.is_bidirectional = False

    lora_config = build_lora_config(args)
    model = DNAGPTForContrastiveLora(
        base,
        lora_config=lora_config,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        mode=args.mode,
    )
    model = model.to(device)
    model.print_trainable_parameters()

    # Override dropout for SimCSE or optional joint view noise.
    if args.mode == "dropout":
        set_dropout(model, args.dropout)
    elif args.view_dropout is not None:
        set_dropout(model, args.view_dropout)

    n_params = sum(p.numel() for p in base.parameters()) / 1e6
    print(f"  Base model : {n_params:.1f}M params")

    # ── 3. Data ───────────────────────────────────────────────────────────────
    print("\n[3/5] Loading data")
    train_dataset, val_dataset = load_contrastive_data(args, tokenizer)
    print(f"  Train : {len(train_dataset):,} sequences")
    print(f"  Val   : {len(val_dataset):,} sequences")

    if args.mode == "dropout":
        collator = DropoutPairCollator(tokenizer=tokenizer, max_length=args.max_length)
    elif args.mode == "revcomp":
        collator = RevCompPairCollator(tokenizer=tokenizer, max_length=args.max_length)
    elif args.mode == "crop":
        _chunk_size = args.chunk_size or args.max_length * 2
        collator = CropPairCollator(
            tokenizer=tokenizer,
            chunk_size=_chunk_size,
            overlap_ratio=args.overlap_ratio,
            max_length=args.max_length,
        )
    else:  # local_shift
        _chunk_size = args.chunk_size or args.max_length * 2
        collator = LocalShiftPairCollator(
            tokenizer=tokenizer,
            chunk_size=_chunk_size,
            max_shift_ratio=args.local_shift_ratio,
            max_length=args.max_length,
        )

    # ── 4. Trainer ────────────────────────────────────────────────────────────
    print("\n[4/5] Building trainer")
    training_args = build_training_args(args)

    metrics_cb = ContrastiveMetricsCallback(
        val_dataset=val_dataset,
        collator=collator,
        model_ref=model,
        args=training_args,
        n_samples=min(256, len(val_dataset)),
    )

    callbacks = [metrics_cb]
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    # Resume
    resume = args.resume
    if resume == "latest":
        resume = find_latest_checkpoint(args.output)
        print(f"  Auto-resume: {resume or 'none found'}")

    trainer = ContrastiveTrainerLora(
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        mode=args.mode,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Training")

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "dna_foundation"
        os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"

    train_result = trainer.train(resume_from_checkpoint=resume)

    # ── Save merged encoder (no projection head) ──────────────────────────────
    print(f"\n  Merging LoRA and saving encoder to: {args.output}")
    model.config.dna_llm2vec_attention_mode = args.attention_mode
    model.config.is_causal = args.attention_mode == "causal"
    model.config.is_bidirectional = args.attention_mode == "bidir"
    model.save_pretrained(args.output, merge=True)
    tokenizer.save_pretrained(args.output)

    print("\n  Training stats:")
    for k, v in train_result.metrics.items():
        print(f"    {k}: {v}")

    print(
        f"\nStep 3 (LoRA) complete.\n"
        f"Encoder saved to: {args.output}\n"
        f"(Projection head discarded use the encoder directly for embeddings.)\n"
        f"Next step: python src/step4_evaluate.py --models 'C0:{args.output}:{args.attention_mode if args.attention_mode == 'causal' else 'bidir'}'"
    )


if __name__ == "__main__":
    main()






