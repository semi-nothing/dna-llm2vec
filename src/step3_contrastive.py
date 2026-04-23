"""
DNA-LLM2Vec  |  Step 3: Contrastive Fine-tuning
=================================================
Trains the MNTP-finetuned bidirectional encoder with a contrastive objective
(InfoNCE / NT-Xent) using two positive-pair strategies:

  --mode dropout   SimCSE baseline — same sequence fed twice through the model
                   with different dropout masks → two views (M3 ablation)

  --mode revcomp   Proposed method — sequence + its reverse complement as the
                   positive pair (biologically equivalent strands) (M4)

Architecture
------------
  Encoder : bidirectional DNAGPT (GPT-2 backbone, Step 1 patch applied)
  Pooling : mean pooling over non-padding token positions
  Head    : optional 2-layer MLP projection head (hidden_dim → proj_dim → proj_dim)
            — head is discarded at save time; only the encoder is kept

Loss
----
  InfoNCE (symmetric NT-Xent) with in-batch negatives.
  temperature τ controls sharpness; default 0.05 follows SimCSE.

Metrics (logged to W&B every eval)
-------
  eval_loss            : mean InfoNCE loss over val set
  eval_alignment       : Wang & Isola (2020) alignment ↓
  eval_uniformity      : Wang & Isola (2020) uniformity ↑ (less negative = better)
  eval_avg_cosine_sim  : mean of off-diagonal cosine similarities

GPU requirements (RTX 5090 / Blackwell sm_120)
-----------------------------------------------
  - CUDA >= 12.8, PyTorch >= 2.7
  - bfloat16 throughout (NOT float16)
  - torch.compile enabled by default (adds ~8-10 min warm-up, then faster)

Usage
-----
  # Dropout pairs (SimCSE baseline):
  uv run python src/step3_contrastive.py \\
      --model  ./mntp_dnagpt \\
      --fasta  ./data/hg38.fa \\
      --output ./contrastive_dnagpt_dropout \\
      --mode   dropout

  # Reverse complement pairs (proposed method):
  uv run python src/step3_contrastive.py \\
      --model  ./mntp_dnagpt \\
      --fasta  ./data/hg38.fa \\
      --output ./contrastive_dnagpt_revcomp \\
      --mode   revcomp

  # Quick smoke-test (no real genome needed):
  uv run python src/step3_contrastive.py \\
      --model ./mntp_dnagpt --output ./contrastive_dnagpt_dropout \\
      --mode dropout --smoke-test

  # Resume:
  uv run python src/step3_contrastive.py \\
      --model ./mntp_dnagpt --fasta ./data/hg38.fa \\
      --output ./contrastive_dnagpt_revcomp --mode revcomp \\
      --resume latest
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
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from step1_bidirectional import patch_to_bidirectional
from data_utils import _iter_fasta, _chunk_sequence


# ── Reverse complement ────────────────────────────────────────────────────────

_RC_TABLE = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (ACGT only)."""
    return seq.translate(_RC_TABLE)[::-1]


# ── Datasets ──────────────────────────────────────────────────────────────────

class RawSequenceDataset(TorchDataset):
    """
    Holds raw DNA strings (not yet tokenised).
    Tokenisation happens in the collator so that the model sees two separate
    forward passes (dropout mode) or two distinct sequences (revcomp mode).
    """

    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


def _smoke_test_sequences(n: int = 2000, min_len: int = 64, max_len: int = 512) -> list[str]:
    """Generate synthetic random DNA sequences for smoke-testing."""
    bases = "ACGT"
    rng = random.Random(42)
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
    Delegates FASTA parsing and chunking to data_utils to avoid code duplication.

    Args:
        filter_n : if True, discard any chunk that contains an N or ambiguous base,
                   matching DNABERT-2 preprocessing. When False (default), non-ACGT
                   chars are stripped before chunking (legacy; creates artificial junctions).
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


# ── Collators ─────────────────────────────────────────────────────────────────

@dataclass
class DropoutPairCollator:
    """
    SimCSE-style collator: tokenises each sequence once and returns
    two identical token tensors. Different dropout masks in the model
    produce two different hidden representations.
    """
    tokenizer: object
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["sequence"] for f in features]
        enc = self.tokenizer(
            seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        # Both views share the same token ids / attention_mask
        return {
            "input_ids_a":      enc["input_ids"],
            "attention_mask_a": enc["attention_mask"],
            "input_ids_b":      enc["input_ids"].clone(),
            "attention_mask_b": enc["attention_mask"].clone(),
        }


@dataclass
class RevCompPairCollator:
    """
    Reverse-complement collator: tokenises seq and RC(seq) separately.
    Positive pairs are biologically equivalent strands.
    """
    tokenizer: object
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["sequence"] for f in features]
        rcs  = [reverse_complement(s) for s in seqs]

        enc_a = self.tokenizer(
            seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            rcs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class MeanPooler(nn.Module):
    """Mean pool transformer hidden states over non-padding positions."""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden       : (B, T, D)
        # attention_mask: (B, T)  — 1 for real tokens, 0 for padding
        # Cast mask to the same dtype as hidden (bfloat16 on CUDA) to avoid
        # silent float32 promotion during the element-wise multiply.
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)  # (B, T, 1)
        sum_hidden = (hidden * mask).sum(dim=1)                      # (B, D)
        lengths    = mask.sum(dim=1).clamp(min=1e-9)                 # (B, 1)
        return sum_hidden / lengths                                  # (B, D)


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head: D → hidden_dim → proj_dim (with LayerNorm).
    Discarded at save time — only the encoder weights are kept.
    """

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


class DNAGPTForContrastive(nn.Module):
    """
    Wraps the MNTP-trained bidirectional DNAGPT for contrastive training.

    Forward pass returns the InfoNCE loss directly so that the HuggingFace
    Trainer's training loop can call model(**batch) and read .loss.
    """

    def __init__(
        self,
        base_model,          # AutoModelForCausalLM (bidirectional-patched GPT-2)
        proj_dim: int = 0,   # 0 → no projection head (use raw pooled embedding)
        temperature: float = 0.05,
        mode: str = "dropout",
    ):
        super().__init__()
        self.gpt2        = base_model.transformer   # GPT-2 backbone
        self.config      = base_model.config
        self.pooler      = MeanPooler()
        self.temperature = temperature
        self.mode        = mode

        hidden_dim = self.config.hidden_size        # 768 for human_gpt2-v1
        if proj_dim > 0:
            self.proj = ProjectionHead(hidden_dim, hidden_dim, proj_dim)
            self.embed_dim = proj_dim
        else:
            self.proj = None
            self.embed_dim = hidden_dim

        # For HuggingFace Trainer compatibility
        self._keys_to_ignore_on_save = None

    # ── Gradient checkpointing (delegated to GPT-2 backbone) ─────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gpt2.gradient_checkpointing_enable(gradient_checkpointing_kwargs or {})

    def gradient_checkpointing_disable(self):
        self.gpt2.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self.gpt2, "gradient_checkpointing", False)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return L2-normalised embeddings of shape (B, embed_dim).
        """
        out    = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state              # (B, T, D)
        pooled = self.pooler(hidden, attention_mask)  # (B, D)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)          # unit vectors

    # ── InfoNCE loss ──────────────────────────────────────────────────────────

    @staticmethod
    def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Symmetric InfoNCE (NT-Xent) loss with in-batch negatives.

        Args:
            z_a, z_b : L2-normalised embeddings, shape (B, D)
            temperature : τ (smaller = sharper, typical 0.05)

        Returns:
            Scalar loss.
        """
        B = z_a.size(0)
        # Cosine similarity matrix — dot product of unit vectors
        sim = torch.mm(z_a, z_b.t()) / temperature   # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(B, device=z_a.device)
        loss_ab = F.cross_entropy(sim,   labels)
        loss_ba = F.cross_entropy(sim.t(), labels)
        return (loss_ab + loss_ba) / 2.0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
        **kwargs,
    ):
        """
        Called by the Trainer with the collator output dict.
        Returns a namespace with .loss so Trainer can call .backward().
        """
        # For dropout mode the two inputs are identical tokens but different
        # dropout masks thanks to model.train() being set by Trainer.
        z_a = self.encode(input_ids_a, attention_mask_a)
        z_b = self.encode(input_ids_b, attention_mask_b)
        loss = self.info_nce_loss(z_a, z_b, self.temperature)

        # Return a simple namespace — Trainer only needs .loss
        return type("ContrastiveOutput", (), {"loss": loss, "z_a": z_a, "z_b": z_b})()

    # ── Save (encoder only) ───────────────────────────────────────────────────

    def save_pretrained(self, save_dir: str):
        """
        Save only the GPT-2 backbone (without projection head) so it can be
        loaded as a standard AutoModelForCausalLM for downstream tasks.

        Device safety:
          - self.gpt2 may live on CUDA; we move it to CPU for saving so that
            GPT2LMHeadModel (created on CPU) and the backbone are on the same
            device when save_pretrained iterates parameters.
          - We call tie_weights() after swapping the transformer so that
            lm_head.weight is re-tied to the new wte.weight (not the stale
            placeholder created by GPT2LMHeadModel.__init__).
          - The backbone is moved back to its original device after saving.
        """
        os.makedirs(save_dir, exist_ok=True)
        from transformers import GPT2LMHeadModel

        # Record and release from CUDA so CPU model can share the same wte
        original_device = next(self.gpt2.parameters()).device
        self.gpt2.to("cpu")

        try:
            full_model = GPT2LMHeadModel(self.config)    # created on CPU
            full_model.transformer = self.gpt2
            # Re-tie lm_head.weight → wte.weight after swapping the transformer
            full_model.tie_weights()
            full_model.save_pretrained(save_dir)
        finally:
            # Always restore backbone to original device (even if save fails)
            self.gpt2.to(original_device)


# ── Custom Trainer ────────────────────────────────────────────────────────────

class ContrastiveTrainer(Trainer):
    """
    Thin subclass of Trainer that:
      - Routes checkpoint saves through DNAGPTForContrastive.save_pretrained()
      - Handles _load_best_model correctly (key-name mismatch with gpt2.*)
      - Disables compute_metrics (loss is sufficient)
    """

    def __init__(self, mode: str, temperature: float, proj_dim: int, **kwargs):
        super().__init__(**kwargs)
        self._contrastive_mode    = mode
        self._contrastive_temp    = temperature
        self._contrastive_projdim = proj_dim

    def _save(self, output_dir: str, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        # Save tokenizer alongside model (transformers 5.x: processing_class)
        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if proc is not None:
            proc.save_pretrained(output_dir)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        ckpt  = self.state.best_model_checkpoint
        dtype  = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        base = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=dtype, attn_implementation="eager"
        )
        base = patch_to_bidirectional(base)
        new_model = DNAGPTForContrastive(
            base,
            proj_dim=self._contrastive_projdim,
            temperature=self._contrastive_temp,
            mode=self._contrastive_mode,
        ).to(device)
        self.model.gpt2   = new_model.gpt2
        self.model.pooler = new_model.pooler
        if new_model.proj is not None:
            self.model.proj = new_model.proj
        print(f"  Loaded best checkpoint: {ckpt}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Evaluation step — compute loss without accumulating logits (avoids OOM).
        """
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss    = outputs.loss.detach()
        return (loss, None, None)


# ── Evaluation metrics callback ───────────────────────────────────────────────

class ContrastiveMetricsCallback(TrainerCallback):
    """
    Computes and logs additional metrics at each evaluation:
      - eval_perplexity  : exp(eval_loss)  — intuitive loss scale
      - eval_alignment   : Wang & Isola (2020) alignment ↓ (lower = tighter pairs)
      - eval_uniformity  : Wang & Isola (2020) uniformity (higher = more spread)
      - eval_avg_cosine_sim : mean off-diagonal similarity (sanity check)
      - tokens_per_sec   : training throughput

    These are computed on a small in-memory sample (first batch of val set)
    to avoid a second full pass over the dataset.
    """

    def __init__(self, val_dataset, collator, tokenizer, model_ref, args, n_samples: int = 256):
        self.val_dataset = val_dataset
        self.collator    = collator
        self.tokenizer   = tokenizer
        self.model_ref   = model_ref   # mutable reference — updated in-place
        self.train_args  = args
        self.n_samples   = n_samples
        self._step_start_time: Optional[float] = None
        self._step_start_step: int = 0
        self._step_start_tokens: int = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self._step_start_time is None:
            self._step_start_time  = time.time()
            self._step_start_step  = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        extra = {}

        # --- Throughput ---
        if "loss" in logs and self._step_start_time is not None:
            elapsed   = time.time() - self._step_start_time
            steps_done = max(state.global_step - self._step_start_step, 1)
            if elapsed > 0:
                extra["train_steps_per_sec"] = round(steps_done / elapsed, 3)
            self._step_start_time = time.time()
            self._step_start_step = state.global_step

        # --- Eval perplexity from scalar loss ---
        if "eval_loss" in logs:
            try:
                extra["eval_perplexity"] = round(math.exp(min(logs["eval_loss"], 20)), 4)
            except OverflowError:
                extra["eval_perplexity"] = float("inf")

            # --- Alignment / uniformity on a small sample ---
            try:
                sample = self._compute_embedding_metrics()
                extra.update(sample)
            except Exception as e:
                print(f"  [ContrastiveMetricsCallback] metric error: {e}")

        if extra and args.report_to and "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.log(extra, step=state.global_step)

        # Print to console
        for k, v in extra.items():
            logs[k] = v

    @torch.no_grad()
    def _compute_embedding_metrics(self) -> dict:
        """
        Run a forward pass on a small val sample and compute:
          alignment, uniformity, avg_cosine_sim
        """
        model = self.model_ref
        model.eval()

        n = min(self.n_samples, len(self.val_dataset))
        indices = list(range(n))
        batch = self.collator([self.val_dataset[i] for i in indices])
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}

        z_a = model.encode(batch["input_ids_a"], batch["attention_mask_a"])
        z_b = model.encode(batch["input_ids_b"], batch["attention_mask_b"])

        # Alignment: mean squared distance between positive pairs (↓ better)
        alignment = (z_a - z_b).pow(2).sum(dim=-1).mean().item()

        # Uniformity: log mean pairwise Gaussian kernel (↑ / less negative = better spread)
        z_all = torch.cat([z_a, z_b], dim=0)
        sq_pdist = torch.pdist(z_all, p=2).pow(2)
        uniformity = torch.log(torch.exp(-2.0 * sq_pdist).mean() + 1e-9).item()

        # Average off-diagonal cosine similarity
        sim_matrix = torch.mm(z_a, z_b.t())
        B = z_a.size(0)
        off_diag = sim_matrix.masked_fill(torch.eye(B, device=z_a.device).bool(), 0.0)
        avg_cos = off_diag.sum() / (B * (B - 1)) if B > 1 else torch.tensor(0.0)

        return {
            "eval_alignment":      round(alignment, 6),
            "eval_uniformity":     round(uniformity, 6),
            "eval_avg_cosine_sim": round(avg_cos.item(), 6),
        }


# ── Training arguments ────────────────────────────────────────────────────────

def build_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=args.grad_ckpt,   # trade ~30% speed for ~60% less activation VRAM
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
        run_name=f"step3_{args.mode}",
        seed=42,
        prediction_loss_only=True,
        remove_unused_columns=False,   # keep "sequence" col for our custom collator
    )


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_contrastive_data(args, tokenizer) -> tuple:
    """
    Return (train_dataset, val_dataset) as RawSequenceDataset instances.
    Tokenisation is deferred to the collator.
    """
    if args.smoke_test:
        print("[Data] Smoke-test mode — generating synthetic sequences")
        seqs = _smoke_test_sequences(n=2000)
        split = int(len(seqs) * 0.99)
        return RawSequenceDataset(seqs[:split]), RawSequenceDataset(seqs[split:])

    if args.fasta:
        print(f"[Data] Loading FASTA: {args.fasta}")
        # BPE compresses ~4 bp/token → bp budget = max_length * 4
        chunk_bp = args.max_length * 4
        stride_bp = args.stride if args.stride > 0 else chunk_bp
        seqs = _extract_sequences_from_fasta(
            args.fasta,
            max_length_bp=chunk_bp,
            stride_bp=stride_bp,
            filter_n=args.filter_n,
        )
        print(f"  Loaded {len(seqs):,} chunks from FASTA")
        random.seed(42)
        random.shuffle(seqs)
        n_val = max(1, int(len(seqs) * args.val_fraction))
        return RawSequenceDataset(seqs[n_val:]), RawSequenceDataset(seqs[:n_val])

    if args.dataset:
        print(f"[Data] Loading HuggingFace dataset: {args.dataset}")
        from datasets import load_dataset
        raw = load_dataset(args.dataset)
        col  = args.sequence_column

        def get_seqs(split):
            return [ex[col] for ex in raw[split] if len(ex[col]) >= 32]

        train_seqs = get_seqs("train")
        if "validation" in raw:
            val_seqs = get_seqs("validation")
        else:
            n_val    = max(1, int(len(train_seqs) * args.val_fraction))
            val_seqs = train_seqs[:n_val]
            train_seqs = train_seqs[n_val:]
        return RawSequenceDataset(train_seqs), RawSequenceDataset(val_seqs)

    raise ValueError("Provide --fasta, --dataset, or --smoke-test")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split("-")[-1]))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DNA-LLM2Vec Step 3: Contrastive Fine-tuning")

    # Model / data
    p.add_argument("--model",  required=True, help="Path to MNTP-trained model (Step 2 output)")
    p.add_argument("--output", required=True, help="Output directory for the contrastive model")
    p.add_argument("--fasta",    default=None, help="Path to .fa or .fa.gz genome file")
    p.add_argument("--dataset",  default=None, help="HuggingFace dataset name")
    p.add_argument("--sequence-column", default="sequence", help="Column name for DNA sequences")
    p.add_argument("--smoke-test", action="store_true", help="Use tiny synthetic data (CI/debug)")

    # Contrastive settings
    p.add_argument("--mode",        default="revcomp", choices=["dropout", "revcomp"],
                   help="Positive pair strategy: dropout (SimCSE) or revcomp (proposed)")
    p.add_argument("--temperature", type=float, default=0.05, help="InfoNCE temperature τ")
    p.add_argument("--proj-dim",    type=int,   default=256,
                   help="Projection head output dim. 0 = no projection head.")

    # Data / tokenisation
    p.add_argument("--max-length",    type=int,   default=1024, help="Max token length per sequence")
    p.add_argument("--stride",        type=int,   default=0,    help="FASTA sliding window stride in bp (0 = non-overlapping)")
    p.add_argument("--filter-n",      action="store_true", default=False,
                   help="Drop FASTA chunks containing any N or ambiguous base "
                        "(DNABERT-2 style). Without this flag, non-ACGT characters "
                        "are stripped in-place, creating artificial junctions.")
    p.add_argument("--val-fraction",  type=float, default=0.01, help="Fraction of data for validation")

    # Training
    p.add_argument("--epochs",     type=int,   default=1,     help="Number of training epochs")
    p.add_argument("--batch-size", type=int,   default=32,    help="Per-device batch size")
    p.add_argument("--grad-accum", type=int,   default=2,     help="Gradient accumulation steps")
    p.add_argument("--lr",         type=float, default=1e-5,  help="Learning rate")
    p.add_argument("--eval-steps", type=int,   default=500,   help="Evaluate every N steps")
    p.add_argument("--save-steps", type=int,   default=500,   help="Save every N steps")
    p.add_argument("--log-steps",  type=int,   default=50,    help="Log every N steps")
    p.add_argument("--compile",    action="store_true", default=False,
                   help="Enable torch.compile (recommended for RTX 5090 after warm-up)")
    p.add_argument("--grad-ckpt",  action="store_true", default=False,
                   help="Enable gradient checkpointing (~30%% slower, ~60%% less activation VRAM)")
    p.add_argument("--patience",   type=int, default=0,
                   help="Early stopping patience in eval steps. 0 = disabled.")

    # Resume / W&B
    p.add_argument("--resume",    default=None,
                   help="Resume from checkpoint path, or 'latest' to auto-detect")
    p.add_argument("--no-wandb",  action="store_true", help="Disable W&B logging")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 60)
    print(f"DNA-LLM2Vec  |  Step 3 — Contrastive ({args.mode})")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mode : {args.mode}  |  τ={args.temperature}  |  proj_dim={args.proj_dim}")
    print("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}")

    # ── Load encoder (bidirectional GPT-2) ────────────────────────────────────
    print(f"\n[2/5] Loading MNTP model from: {args.model}")
    # Do NOT use device_map="auto" here. device_map wraps the model in an
    # Accelerate dispatch object, which conflicts with the Trainer's own device
    # placement logic and breaks when we extract .transformer into
    # DNAGPTForContrastive. Load on CPU; Trainer moves the wrapped model to GPU.
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    base = patch_to_bidirectional(base)
    n_params = sum(p.numel() for p in base.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")

    # ── Wrap in contrastive model ──────────────────────────────────────────────
    model = DNAGPTForContrastive(
        base,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        mode=args.mode,
    )
    model = model.to(device)

    # ── Resume: reload encoder weights ────────────────────────────────────────
    resume = args.resume
    if resume == "latest":
        resume = find_latest_checkpoint(args.output)
        if resume:
            print(f"  Auto-detected checkpoint: {resume}")
        else:
            print("  No existing checkpoint found — starting fresh")
            resume = None

    if resume:
        print(f"\n  Reloading encoder weights from: {resume}")
        ckpt_base = AutoModelForCausalLM.from_pretrained(
            resume, torch_dtype=dtype, attn_implementation="eager"
        )
        ckpt_base = patch_to_bidirectional(ckpt_base)
        new_wrap  = DNAGPTForContrastive(
            ckpt_base,
            proj_dim=args.proj_dim,
            temperature=args.temperature,
            mode=args.mode,
        )
        model.gpt2   = new_wrap.gpt2.to(next(model.parameters()).device)
        model.pooler = new_wrap.pooler
        if new_wrap.proj is not None and model.proj is not None:
            model.proj.load_state_dict(new_wrap.proj.state_dict())
        print("  Encoder weights reloaded.")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[3/5] Loading data")
    train_dataset, val_dataset = load_contrastive_data(args, tokenizer)
    print(f"  Train : {len(train_dataset):,} sequences")
    print(f"  Val   : {len(val_dataset):,} sequences")

    # ── Collator ──────────────────────────────────────────────────────────────
    if args.mode == "dropout":
        collator = DropoutPairCollator(tokenizer=tokenizer, max_length=args.max_length)
    else:
        collator = RevCompPairCollator(tokenizer=tokenizer, max_length=args.max_length)

    # ── Training arguments ────────────────────────────────────────────────────
    print("\n[4/5] Building training arguments")
    training_args = build_training_args(args)

    # ── Metrics callback ──────────────────────────────────────────────────────
    metrics_cb = ContrastiveMetricsCallback(
        val_dataset=val_dataset,
        collator=collator,
        tokenizer=tokenizer,
        model_ref=model,
        args=training_args,
        n_samples=min(256, len(val_dataset)),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = ContrastiveTrainer(
        mode=args.mode,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[metrics_cb] + (
            [EarlyStoppingCallback(early_stopping_patience=args.patience)]
            if args.patience > 0 else []
        ),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Training")
    effective_batch = (
        args.batch_size
        * args.grad_accum
        * max(1, torch.cuda.device_count() if device == "cuda" else 1)
    )
    print(f"  Effective batch size : {effective_batch}")
    print(f"  Mode                 : {args.mode}")
    print(f"  Temperature τ        : {args.temperature}")
    print(f"  Projection dim       : {args.proj_dim or 'none (raw embedding)'}")

    # Set W&B project and entity before training starts
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "dna_foundation"
        os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"

    train_result = trainer.train(resume_from_checkpoint=resume)

    # ── Save final encoder ────────────────────────────────────────────────────
    print(f"\n  Saving encoder to: {args.output}")
    trainer.save_model(args.output)

    print("\n  Training stats:")
    for k, v in train_result.metrics.items():
        print(f"    {k}: {v}")

    print(
        f"\nStep 3 complete.\n"
        f"Encoder saved to: {args.output}\n"
        f"(Projection head was NOT saved — use the encoder directly for embeddings.)"
    )


if __name__ == "__main__":
    main()
