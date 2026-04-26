"""
DNA-LLM2Vec  |  Step 2: Masked Next Token Prediction (MNTP) Fine-tuning
========================================================================
Fine-tunes the bidirectional DNAGPT model (from Step 1) using a masked
language modelling (MLM) objective. The existing LM head is reused as
the prediction head — no new parameters are added.

GPU requirements (RTX 5090 / Blackwell sm_120):
  - PyTorch >= 2.7, CUDA >= 12.8
  - bfloat16 is used throughout (NOT float16)

Usage:
  # From a FASTA file (recommended for full genome pretraining):
  python src/step2_mntp.py \\
      --model  ./bidir_dnagpt \\
      --fasta  /path/to/hg38.fa \\
      --output ./mntp_dnagpt

  # From a HuggingFace dataset:
  python src/step2_mntp.py \\
      --model   ./bidir_dnagpt \\
      --dataset dnagpt/human_genome \\
      --output  ./mntp_dnagpt

  # Quick smoke-test (tiny synthetic data, no real genome needed):
  python src/step2_mntp.py --model ./bidir_dnagpt --output ./mntp_dnagpt --smoke-test
"""

import argparse
import math
import os
import sys
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import MaskedLMOutput

# Local utilities
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_fasta_dataset, load_hub_dataset, ensure_mask_token
from step1_bidirectional import patch_to_bidirectional


# ── Model wrapper: GPT-2 decoder → MLM ───────────────────────────────────────

class DNAGPTForMNTP(nn.Module):
    """
    Wraps a bidirectional GPT-2 model for Masked Next Token Prediction.

    Loss (LLM2Vec-style, matching step2_mntp_lora.py):
      logits[:, :-1] paired with labels[:, 1:]
      - at each position i-1, predict the original token at masked position i.
      - labels are -100 at non-masked positions (ignored)

    The existing lm_head from GPT-2 (Wte transpose or linear) is reused.
    No new parameters are added. Weight tying (lm_head.weight == wte.weight)
    is preserved — updates to the output projection also update the input
    embeddings, consistent with LLM2Vec and standard GPT-2 fine-tuning.
    """

    _keys_to_ignore_on_save = None   # expected by Trainer._issue_warnings_after_load

    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.gpt2    = base_model.transformer   # GPT2Model (no LM head)
        self.lm_head = base_model.lm_head       # Linear: hidden_size → vocab_size
        self.config  = base_model.config
        # Weight tying is preserved: lm_head.weight IS wte.weight.
        # Serialisation is handled by save_safetensors=False in TrainingArguments,
        # which uses torch.save (supports shared tensors) for checkpoints.

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        """
        Args:
            input_ids      : (B, T)  — masked input sequence
            attention_mask : (B, T)  — 1 for real tokens, 0 for padding
            labels         : (B, T)  — original token ids at masked positions,
                                       -100 everywhere else (ignored in loss)
        """
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state          # (B, T, D)
        logits = self.lm_head(hidden)               # (B, T, V)

        loss = None
        if labels is not None:
            # LLM2Vec shift: predict token at position i from hidden state at i-1.
            # Drop the last logit (no target) and the first label (no predictor).
            logits_s = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            labels_s = labels[:, 1:].contiguous()        # (B, T-1)
            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels_s.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_dir: str):
        """
        Reconstruct the full GPT2LMHeadModel and save it, so Step 3 can
        load with load_bidirectional_dnagpt() as normal.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Re-attach lm_head to the transformer to form a complete model
        # We borrow from AutoModelForCausalLM's internal structure
        from transformers import GPT2LMHeadModel
        full_model = GPT2LMHeadModel(self.config)
        full_model.transformer = self.gpt2
        full_model.lm_head     = self.lm_head
        full_model.save_pretrained(save_dir)


# ── Smoke-test dataset ────────────────────────────────────────────────────────

class SyntheticDNADataset(TorchDataset):
    """
    Tiny synthetic dataset for quick smoke-tests.
    Generates random ACGT sequences, tokenises them, returns dict of tensors.
    No real genome data needed.
    """
    BASES = ["A", "C", "G", "T"]

    def __init__(self, tokenizer, n_samples: int = 256, seq_len: int = 128):
        self.samples = []
        for _ in range(n_samples):
            seq = "".join(random.choices(self.BASES, k=seq_len * 4))
            enc = tokenizer(
                seq,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt",
                return_special_tokens_mask=True,   # needed by DataCollatorForLanguageModeling
            )
            self.samples.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics from model logits and labels.

    Called by the Trainer after every eval step. Receives:
      eval_pred.predictions : np.ndarray  (N, T, V)  — raw logits
      eval_pred.label_ids   : np.ndarray  (N, T)     — -100 at non-masked positions

    Returns a dict logged to W&B under eval/:
      perplexity     — exp(cross-entropy loss on masked tokens only)
      accuracy_top1  — fraction of masked tokens predicted correctly (top-1)
      accuracy_top5  — fraction of masked tokens in top-5 predictions
      n_masked       — total masked tokens evaluated (sanity check)
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Match the training objective: logits at i-1 predict labels at i.
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # Identify masked positions (label != -100)
    mask = labels != -100                                   # (N, T) bool

    # Perplexity from cross-entropy on masked tokens only
    logits_t  = torch.tensor(logits,  dtype=torch.float32)
    labels_t  = torch.tensor(labels,  dtype=torch.long)
    ce_loss = F.cross_entropy(
        logits_t.view(-1, logits_t.size(-1)),
        labels_t.view(-1),
        ignore_index=-100,
        reduction="mean",
    ).item()
    perplexity = math.exp(min(ce_loss, 20))                 # cap at e^20 to avoid inf

    # Top-1 accuracy
    preds_top1 = np.argmax(logits, axis=-1)                 # (N, T)
    correct_top1 = (preds_top1 == labels) & mask
    acc_top1 = correct_top1.sum() / mask.sum()

    # Top-5 accuracy
    top5_indices = np.argpartition(logits, -5, axis=-1)[..., -5:]  # (N, T, 5)
    labels_expanded = labels[..., np.newaxis]               # (N, T, 1)
    correct_top5 = ((top5_indices == labels_expanded) & mask[..., np.newaxis]).any(axis=-1)
    acc_top5 = correct_top5.sum() / mask.sum()

    return {
        "perplexity":    round(perplexity, 4),
        "accuracy_top1": round(float(acc_top1), 4),
        "accuracy_top5": round(float(acc_top5), 4),
        "n_masked":      int(mask.sum()),
    }


class ThroughputCallback(TrainerCallback):
    """
    Logs tokens/sec and samples/sec to W&B at every logging step.
    Also logs train perplexity (exp(loss)) alongside the raw loss.
    """

    def __init__(self, max_length: int):
        self.max_length  = max_length   # tokens per sample
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        extra = {}

        # Train perplexity from the step loss.
        # Normalize by grad_accum_steps: transformers 5.x accumulates raw step
        # losses before dividing by optimizer steps, so the logged loss is
        # scaled by gradient_accumulation_steps. Dividing corrects this so
        # runs with different accum settings are directly comparable in W&B.
        if "loss" in logs:
            normalized_loss = logs["loss"] / args.gradient_accumulation_steps
            try:
                extra["train_loss_normalized"] = round(normalized_loss, 4)
                extra["train_perplexity"]      = round(math.exp(min(normalized_loss, 20)), 4)
            except OverflowError:
                extra["train_perplexity"] = float("inf")

        # Eval perplexity from the eval loss (no logits needed)
        if "eval_loss" in logs:
            try:
                extra["eval_perplexity"] = round(math.exp(min(logs["eval_loss"], 20)), 4)
            except OverflowError:
                extra["eval_perplexity"] = float("inf")

        # Throughput — only meaningful if we have step timing
        if self._step_start is not None:
            elapsed = time.perf_counter() - self._step_start
            if elapsed > 0:
                batch_tokens  = args.per_device_train_batch_size * self.max_length
                extra["tokens_per_sec"]  = round(batch_tokens / elapsed)
                extra["samples_per_sec"] = round(args.per_device_train_batch_size / elapsed, 1)

        if extra and args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(extra, step=state.global_step)
            except Exception:
                pass


# ── Custom Trainer ────────────────────────────────────────────────────────────

class MNTPTrainer(Trainer):
    """
    Trainer subclass with two overrides needed for DNAGPTForMNTP:

    1. _save — routes checkpoint saves through DNAGPTForMNTP.save_pretrained(),
       which reconstructs a full GPT2LMHeadModel and uses its own save logic.
       This is required because GPT-2 ties lm_head.weight to wte.weight, and
       safetensors rejects shared tensors in plain nn.Module wrappers.

    2. _load_best_model — reloads the best checkpoint back into the wrapper
       using AutoModelForCausalLM + patch_to_bidirectional, matching the key
       names expected by DNAGPTForMNTP (gpt2.* / lm_head.*).
    """

    def _save(self, output_dir: str, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        ckpt = self.state.best_model_checkpoint
        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        base = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=dtype, attn_implementation="eager"
        )
        base = patch_to_bidirectional(base)
        new_wrapper = DNAGPTForMNTP(base).to(device)
        self.model.gpt2    = new_wrapper.gpt2
        self.model.lm_head = new_wrapper.lm_head
        print(f"  Loaded best checkpoint from: {ckpt}")


# ── Training setup ────────────────────────────────────────────────────────────

def build_training_args(output_dir: str, args: argparse.Namespace) -> TrainingArguments:
    """
    TrainingArguments tuned for RTX 5090 (Blackwell sm_120).

    Key 5090-specific choices:
      - bf16=True          : Blackwell has native bf16 tensor cores;
                             fp16 can cause instability on sm_120
      - bf16_full_eval     : evaluate in bf16 to stay consistent
      - torch_compile=False: PyTorch 2.7 Blackwell compile support is
                             improving but still has edge cases; disable
                             for stability, enable manually once verified
      - dataloader_pin_memory=True: speeds up CPU→GPU transfers on PCIe 5.0
    """
    return TrainingArguments(
        output_dir=output_dir,

        # ── Precision ──────────────────────────────────────────────────────
        bf16=True,
        bf16_full_eval=True,
        # fp16=False  (never use fp16 on Blackwell)

        # ── Batch size & gradient accumulation ────────────────────────────
        # Effective batch = per_device * gradient_accumulation_steps
        # With 32 GB VRAM on 5090, per_device=16 is conservative for 512-token seqs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # ── Optimiser ─────────────────────────────────────────────────────
        learning_rate=args.lr,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,

        # ── Schedule ──────────────────────────────────────────────────────
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",

        # ── Logging & saving ──────────────────────────────────────────────
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        eval_strategy="steps",         # renamed from evaluation_strategy in transformers>=4.46
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── DataLoader ────────────────────────────────────────────────────
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # ── Misc ──────────────────────────────────────────────────────────
        seed=args.seed,
        run_name=args.run_name or (
            f"step2_mntp_s{args.seed}"
            + (f"_r{args.repeat_index}" if args.repeat_index is not None else "")
        ),
        report_to="wandb",
        torch_compile=True,      # PyTorch 2.11 has stable Blackwell (sm_120) support
        remove_unused_columns=False,
        gradient_checkpointing=False,  # set True to trade ~30% speed for ~60% less activation VRAM
        prediction_loss_only=True,     # avoids accumulating full logit tensors in VRAM
                                       # (hg38 val set: ~14K samples × 512 × 25K vocab = OOM)
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DNA-LLM2Vec Step 2: MNTP")

    # Model
    parser.add_argument(
        "--model", default="./bidir_dnagpt",
        help="Path to bidirectional model from Step 1 (default: ./bidir_dnagpt)",
    )
    parser.add_argument(
        "--output", default="./mntp_dnagpt",
        help="Output directory for the MNTP-fine-tuned model",
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--fasta", default=None,
        help="Path to genome FASTA file (e.g. hg38.fa or hg38.fa.gz)",
    )
    data_group.add_argument(
        "--dataset", default=None,
        help="HuggingFace dataset name (e.g. dnagpt/human_genome)",
    )
    data_group.add_argument(
        "--smoke-test", action="store_true",
        help="Run a quick smoke-test with synthetic data (no genome file needed)",
    )

    # Data options
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max token length per training sample (default: 1024)")
    parser.add_argument("--stride", type=int, default=0,
                        help="Sliding window stride in bp for FASTA chunking. "
                             "0 = non-overlapping windows (default, memory-safe for hg38). "
                             "Set to e.g. 1024 for 50%% overlap.")
    parser.add_argument("--filter-n", action="store_true", default=False,
                        help="Drop FASTA chunks that contain any N or ambiguous base, "
                             "matching DNABERT-2 preprocessing. Without this flag, "
                             "non-ACGT characters are silently removed before chunking "
                             "(creates artificial junctions across chromosomal N-gaps).")
    parser.add_argument("--mlm-probability", type=float, default=0.15,
                        help="Fraction of tokens to mask (default: 0.15)")

    # Training
    parser.add_argument("--epochs",      type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--max-steps",   type=int,   default=-1,
                        help="Maximum optimizer steps. -1 disables the limit and uses --epochs only.")
    parser.add_argument("--batch-size",  type=int,   default=16,   help="Per-device batch size (default: 16)")
    parser.add_argument("--grad-accum",  type=int,   default=2,    help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr",          type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmup-steps",type=int,   default=500,  help="LR warmup steps (default: 500)")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument("--run-name",    default=None,
                        help="Optional W&B run name. If omitted, a descriptive name is generated.")
    parser.add_argument("--repeat-index", type=int, default=None,
                        help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    parser.add_argument(
        "--resume", default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training from a checkpoint directory, e.g. "
             "./mntp_dnagpt/checkpoints/checkpoint-500. "
             "Pass 'latest' to auto-detect the most recent checkpoint.",
    )

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("DNA-LLM2Vec  |  Step 2: MNTP Fine-tuning")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max steps: {args.max_steps}")
        print(f"  dtype : bfloat16")
    print(f"  Model : {args.model}")
    print(f"  Output: {args.output}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print("=" * 60)

    # ── 1. Load tokenizer ────────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load bidirectional model ──────────────────────────────────────────
    print("\n[2/5] Loading bidirectional model (from Step 1)")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load on CPU first — do NOT use device_map="auto" here.
    # device_map wraps the model in an Accelerate dispatch object, which breaks
    # when we extract .transformer and .lm_head into DNAGPTForMNTP and then
    # pass the wrapper to Trainer (Trainer moves the model to device itself).
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",   # must match Step 1 — keeps bias buffer patchable
    )

    # Re-apply bidirectional patch (bias buffer is non-persistent)
    base_model = patch_to_bidirectional(base_model)

    # Add [MASK] token if the GPT-2 tokenizer lacks one
    ensure_mask_token(tokenizer, base_model)

    # Wrap in MNTP model (replaces causal LM loss with MLM loss)
    model = DNAGPTForMNTP(base_model)
    # Move to GPU — Trainer will keep it there
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")

    # ── 3. Load dataset ──────────────────────────────────────────────────────
    print("\n[3/5] Preparing dataset")

    if args.smoke_test:
        print("  Mode: synthetic smoke-test data")
        train_dataset = SyntheticDNADataset(tokenizer, n_samples=256, seq_len=args.max_length)
        eval_dataset  = SyntheticDNADataset(tokenizer, n_samples=64,  seq_len=args.max_length)

    elif args.fasta:
        print(f"  Mode: FASTA file — {args.fasta}")
        datasets = load_fasta_dataset(
            args.fasta, tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            filter_n=args.filter_n,
        )
        train_dataset = datasets["train"]
        eval_dataset  = datasets["validation"]

    elif args.dataset:
        print(f"  Mode: HuggingFace dataset — {args.dataset}")
        datasets = load_hub_dataset(args.dataset, tokenizer, max_length=args.max_length)
        train_dataset = datasets["train"]
        eval_dataset  = datasets.get("validation", None)

    else:
        print("  No data source specified. Use --fasta, --dataset, or --smoke-test.")
        print("  Run with --smoke-test to verify the pipeline quickly.")
        sys.exit(1)

    print(f"  Train samples : {len(train_dataset):,}")
    if eval_dataset:
        print(f"  Val   samples : {len(eval_dataset):,}")

    # ── 4. Data collator (handles masking at runtime) ────────────────────────
    print("\n[4/5] Setting up training")

    # DataCollatorForLanguageModeling applies the MLM masking on-the-fly:
    #   - 80% of selected tokens → [MASK]
    #   - 10% → random token from vocabulary
    #   - 10% → unchanged (kept as-is)
    # Labels are set to -100 at non-selected positions (ignored in loss).
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = build_training_args(
        output_dir=os.path.join(args.output, "checkpoints"),
        args=args,
    )

    trainer = MNTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,    # renamed from tokenizer= in transformers>=4.46
        callbacks=[ThroughputCallback(max_length=args.max_length)],
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("\n[5/5] Training")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size} × {args.grad_accum} accum steps")
    print(f"  Effective batch   : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate     : {args.lr}")
    print(f"  MLM probability   : {args.mlm_probability}")
    print("  Pred objective    : LLM2Vec i-1 shift (logits[:, :-1] -> labels[:, 1:])")
    print()

    # Resolve 'latest' to the most recent checkpoint in the checkpoints dir
    resume = args.resume
    if resume == "latest":
        ckpt_dir = os.path.join(args.output, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpts = sorted(
                [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]),
            )
            resume = os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None
        else:
            resume = None
        print(f"  Resuming from : {resume or 'none found — starting fresh'}")

    if resume:
        # The Trainer restores optimizer/scheduler/RNG state from the checkpoint dir,
        # but uses load_state_dict() with GPT2LMHeadModel key names (transformer.*)
        # while DNAGPTForMNTP uses gpt2.* — keys mismatch, so weights silently fail.
        # We manually reload model weights here; Trainer handles the rest.
        print(f"\n  Loading model weights from checkpoint: {resume}")
        base_ckpt = AutoModelForCausalLM.from_pretrained(
            resume, torch_dtype=dtype, attn_implementation="eager"
        )
        base_ckpt = patch_to_bidirectional(base_ckpt)
        ensure_mask_token(tokenizer, base_ckpt)
        new_wrapper = DNAGPTForMNTP(base_ckpt).to(device)
        model.gpt2    = new_wrapper.gpt2
        model.lm_head = new_wrapper.lm_head
        print("  Model weights restored from checkpoint.")

    # Set W&B project and entity before training starts
    os.environ["WANDB_PROJECT"] = "dna_foundation"
    os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"

    train_result = trainer.train(resume_from_checkpoint=resume)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving MNTP model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nStep 2 complete.")
    print(f"  Train loss : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Next step    : python src/step3_contrastive.py --model {args.output}")


if __name__ == "__main__":
    main()
