"""
DNA-LLM2Vec  |  Step 3 (NT-500M): Reverse Complement Contrastive Fine-tuning
=============================================================================
Applies revcomp contrastive learning (LoRA) to Nucleotide Transformer 500M.

NT-500M is already a bidirectional encoder pre-trained with MLM on DNA:
  - Step 1 (bidirectional patch): SKIP — already an encoder
  - Step 2 (MNTP): SKIP — already pre-trained with MLM
  - Step 3 (revcomp contrastive): applied here with LoRA

6-mer tokenization makes revcomp contrastive cleaner than BPE:
  revcomp(6-mer) maps to exactly one token — no BPE asymmetry.

Chunk size: 3000 nt → 500 6-mers + 2 special tokens = 502 tokens (< 512 limit).

Usage:
  uv run python src/step3_contrastive_nt.py \\
      --model InstaDeepAI/nucleotide-transformer-500m-human-ref \\
      --fasta ./data/hg38.fa \\
      --output ./contrastive_nt500m_revcomp \\
      --max-steps 1000

  # Smoke test:
  uv run python src/step3_contrastive_nt.py \\
      --model InstaDeepAI/nucleotide-transformer-500m-human-ref \\
      --fasta ./data/hg38.fa --output ./test_nt --smoke-test
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import _iter_fasta, _chunk_sequence


# ── Reverse complement ────────────────────────────────────────────────────────

_RC_TABLE = str.maketrans("ACGT", "TGCA")

def reverse_complement(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class RevCompDataset(TorchDataset):
    """Streams ACGT-only chunks from hg38.fa for revcomp contrastive training."""

    def __init__(self, fasta_path: str, chunk_size: int, max_samples: int = 500_000):
        self.chunks = []
        print(f"  Reading {fasta_path} ...", flush=True)
        for seq in _iter_fasta(fasta_path):
            for chunk in _chunk_sequence(seq, chunk_size, stride=chunk_size):
                if set(chunk.upper()) <= {"A", "C", "G", "T"}:
                    self.chunks.append(chunk.upper())
                if len(self.chunks) >= max_samples:
                    break
            if len(self.chunks) >= max_samples:
                break
        random.shuffle(self.chunks)
        print(f"  {len(self.chunks):,} chunks loaded", flush=True)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq = self.chunks[idx]
        return {"seq": seq, "seq_rc": reverse_complement(seq)}


# ── Subsequence Crop Dataset ──────────────────────────────────────────────────

class SubseqCropDataset(TorchDataset):
    """Positive pairs = two overlapping crops from the same genomic window.

    Biological assumption: different overlapping fragments of the same regulatory
    region (enhancer / promoter) share functional context.

    Window layout stored per sample (long_size = chunk_size + max_shift):
      |<────────────────── long_size ──────────────────>|
      |<── crop A ──────────>|                          |
      |          |<── crop B ──────────>|               |
      |←a_start→|←b_start→|                            |
                 |←overlap→|

    Overlap ≥ min_overlap is guaranteed when |b_start - a_start| ≤ max_shift.
    """

    def __init__(self, fasta_path: str, chunk_size: int,
                 min_overlap: int | None = None, max_samples: int = 500_000):
        self.chunk_size  = chunk_size
        self.min_overlap = min_overlap if min_overlap is not None else chunk_size // 2
        if not (0 < self.min_overlap < chunk_size):
            raise ValueError("min_overlap must be in (0, chunk_size)")
        self.max_shift = chunk_size - self.min_overlap   # max |b_start - a_start|
        long_size      = chunk_size + self.max_shift     # window stored per sample

        self.seqs = []
        print(f"  Reading {fasta_path}  [crop mode, long_size={long_size} nt, "
              f"min_overlap={self.min_overlap} nt] ...", flush=True)
        for seq in _iter_fasta(fasta_path):
            for chunk in _chunk_sequence(seq, long_size, stride=long_size):
                if len(chunk) == long_size and set(chunk.upper()) <= {"A", "C", "G", "T"}:
                    self.seqs.append(chunk.upper())
                if len(self.seqs) >= max_samples:
                    break
            if len(self.seqs) >= max_samples:
                break
        random.shuffle(self.seqs)
        print(f"  {len(self.seqs):,} long windows loaded", flush=True)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq     = self.seqs[idx]
        # Both start positions in [0, max_shift] → guaranteed overlap ≥ min_overlap
        a_start = random.randint(0, self.max_shift)
        b_start = random.randint(0, self.max_shift)
        return {
            "seq":    seq[a_start : a_start + self.chunk_size],
            "seq_rc": seq[b_start : b_start + self.chunk_size],   # reuses collator key
        }


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class RevCompCollator:
    tokenizer: object
    max_length: int

    def __call__(self, batch):
        seqs    = [x["seq"]    for x in batch]
        seqs_rc = [x["seq_rc"] for x in batch]
        enc_a = self.tokenizer(seqs,    truncation=True, max_length=self.max_length,
                               padding="longest", return_tensors="pt")
        enc_b = self.tokenizer(seqs_rc, truncation=True, max_length=self.max_length,
                               padding="longest", return_tensors="pt")
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class NTContrastiveModel(nn.Module):
    """NT-500M encoder + LoRA adapters for revcomp contrastive learning."""

    def __init__(self, peft_model, temperature: float = 0.05):
        super().__init__()
        self.peft_model  = peft_model
        self.temperature = temperature

    def _encode(self, input_ids, attention_mask):
        """Mean-pool encoder output → L2-normalised embedding."""
        out    = self.peft_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                              # (B, T, D)
        mask   = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)                          # (B, D)

    def forward(self, input_ids_a, attention_mask_a,
                      input_ids_b, attention_mask_b):
        z_a = self._encode(input_ids_a, attention_mask_a)
        z_b = self._encode(input_ids_b, attention_mask_b)

        # Symmetric InfoNCE (NT-Xent style): positives on diagonal
        sim    = torch.matmul(z_a, z_b.T) / self.temperature       # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)
        loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    # ── Gradient checkpointing ────────────────────────────────────────────────
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.peft_model.enable_input_require_grads()
        kwargs = {"use_reentrant": False}
        if gradient_checkpointing_kwargs:
            kwargs.update(gradient_checkpointing_kwargs)
        # Calls gradient_checkpointing_enable on the underlying EsmModel
        self.peft_model.base_model.model.gradient_checkpointing_enable(kwargs)

    def gradient_checkpointing_disable(self):
        self.peft_model.base_model.model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self.peft_model.base_model.model, "gradient_checkpointing", False)


# ── Custom Trainer ────────────────────────────────────────────────────────────

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model(
            input_ids_a=inputs["input_ids_a"],
            attention_mask_a=inputs["attention_mask_a"],
            input_ids_b=inputs["input_ids_b"],
            attention_mask_b=inputs["attention_mask_b"],
        )
        return (loss, loss) if return_outputs else loss

    def _save(self, output_dir, state_dict=None):
        """Make all tensors contiguous before saving (required for bfloat16 + LoRA)."""
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        super()._save(output_dir, state_dict=state_dict)


# ── Throughput callback ───────────────────────────────────────────────────────

class ThroughputCallback(TrainerCallback):
    def __init__(self, batch_size: int, grad_accum: int):
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self._t0        = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t0 is None:
            self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self._t0 is not None:
            elapsed  = time.time() - self._t0
            steps    = state.global_step
            samples  = steps * self.batch_size * self.grad_accum
            print(f"  [step {steps}] loss={logs.get('loss', float('nan')):.4f}  "
                  f"samples/s={samples/elapsed:.1f}  "
                  f"steps/s={steps/elapsed:.3f}", flush=True)


# ── LoRA config ───────────────────────────────────────────────────────────────

def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """LoRA for ESM/BERT-style encoder (standard nn.Linear, no fan_in_fan_out needed)."""
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "value"],   # ESM self-attention projections
        bias="none",
    )


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DNA-LLM2Vec Step 3 (NT-500M): Revcomp Contrastive Fine-tuning"
    )
    p.add_argument("--model",       default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                   help="HuggingFace model ID or local path for NT-500M.")
    p.add_argument("--fasta",       required=True,
                   help="Path to hg38.fa (or any reference genome FASTA).")
    p.add_argument("--output",      required=True,
                   help="Output directory for merged model + tokenizer.")
    p.add_argument("--mode",        default="revcomp", choices=["revcomp", "crop"],
                   help="Contrastive mode: 'revcomp' (reverse complement pairs) or "
                        "'crop' (overlapping subsequence crops from the same region).")
    p.add_argument("--overlap-ratio", type=float, default=0.5,
                   help="[crop mode] Minimum overlap between the two crops as a fraction "
                        "of chunk_size (0 < ratio < 1). Default: 0.5 (50%% overlap).")
    p.add_argument("--max-steps",   type=int, default=1000,
                   help="Max training steps (-1 = use --epochs instead).")
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--grad-accum",  type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = batch-size × grad-accum.")
    p.add_argument("--max-length",  type=int, default=512,
                   help="Max token length. 512 tokens = 3000 nt with 6-mer tokenization.")
    p.add_argument("--chunk-size",  type=int, default=3000,
                   help="Nucleotide chunk size from hg38.fa. 3000 nt → 500 6-mers → 502 tokens.")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup",      type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lora-r",      type=int, default=16)
    p.add_argument("--lora-alpha",  type=int, default=32)
    p.add_argument("--lora-dropout",type=float, default=0.1)
    p.add_argument("--grad-ckpt",   action="store_true",
                   help="Enable gradient checkpointing (saves memory, ~30% slower).")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick 10-step run to verify the pipeline works.")
    p.add_argument("--no-wandb",    action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    if args.smoke_test:
        args.max_steps  = 10
        args.batch_size = 4

    min_overlap = int(args.chunk_size * args.overlap_ratio)
    mode_label = {
        "revcomp": "revcomp — reverse complement pairs",
        "crop":    f"crop — overlapping subseq crops "
                   f"(overlap_ratio={args.overlap_ratio:.0%}, min_overlap={min_overlap} nt)",
    }[args.mode]

    print("=" * 68)
    print("DNA-LLM2Vec  |  Step 3 (NT-500M) — Contrastive Fine-tuning")
    print("=" * 68)
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Model       : {args.model}")
    print(f"  Mode        : {mode_label}")
    print(f"  Max steps   : {args.max_steps}")
    eff_batch = args.batch_size * args.grad_accum
    print(f"  Batch size  : {args.batch_size}  |  Grad accum: {args.grad_accum}"
          f"  (effective: {eff_batch})")
    print(f"  Max length  : {args.max_length} tokens  ({args.chunk_size} nt)")
    print(f"  LoRA        : r={args.lora_r}  alpha={args.lora_alpha}"
          f"  dropout={args.lora_dropout}")
    print(f"  Grad ckpt   : {args.grad_ckpt}")
    print("=" * 68)

    # ── W&B ──────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "dna_foundation"
        os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"
        try:
            import wandb
            run_name = f"step3_nt500m_{args.mode}"
            wandb.init(job_type="contrastive", name=run_name, config=vars(args))
        except Exception as e:
            print(f"  [warn] W&B init failed: {e}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(args.model, dtype=dtype)
    base.eval()
    n_params = sum(p.numel() for p in base.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    print("\n[2/4] Applying LoRA")
    lora_cfg   = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    peft_model = get_peft_model(base, lora_cfg)
    peft_model.print_trainable_parameters()

    model = NTContrastiveModel(peft_model, temperature=args.temperature)
    model = model.to(device)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED (use_reentrant=False)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n[3/4] Loading dataset")
    if args.mode == "revcomp":
        dataset = RevCompDataset(args.fasta, chunk_size=args.chunk_size)
    else:  # crop
        dataset = SubseqCropDataset(
            args.fasta,
            chunk_size=args.chunk_size,
            min_overlap=min_overlap,
        )
    collator = RevCompCollator(tokenizer=tokenizer, max_length=args.max_length)

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[4/4] Training")
    train_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 100,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="wandb" if not args.no_wandb else "none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[ThroughputCallback(args.batch_size, args.grad_accum)],
    )
    trainer.train()

    # ── Save merged model ─────────────────────────────────────────────────────
    print(f"\nMerging LoRA adapters → {args.output}")
    merged = model.peft_model.merge_and_unload()
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to: {args.output}")

    if not args.no_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
