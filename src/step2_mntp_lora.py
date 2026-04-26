"""
DNA-LLM2Vec  |  Step 2 (LoRA): Masked Next Token Prediction — LLM2Vec style
=============================================================================
LoRA-based MNTP with two key improvements over step2_mntp.py:

  1. LoRA (PEFT): only adapter weights (~1% of params) are updated.
     Base model weights are frozen, reducing catastrophic forgetting
     and saving activation VRAM compared to full fine-tuning.

  2. LLM2Vec prediction position (BehnamGhader et al., 2024, Figure 1):
     masked token at position i is predicted from the hidden state at
     position i-1, not from the [MASK] position itself.

     Standard MLM : logits[:, i]   → predict labels[:, i]
     LLM2Vec MNTP : logits[:, i-1] → predict labels[:, i]
                    ≡  logits[:, :-1]  paired with  labels[:, 1:]

     Rationale: GPT-2 was pretrained to predict token[i+1] from position[i].
     Predicting the masked token from position i-1 keeps the loss objective
     consistent with pretraining while using full bidirectional context.

Mask token:
  Unlike step2_mntp.py, this version avoids resizing the embedding table.
  Instead, the EOS token is reused as the mask token (analogous to LLM2Vec
  using underscore '_' for LLaMA/Mistral). This sidesteps weight-tying
  complications between wte and lm_head when applying LoRA.

Checkpoint layout:
  <output>/checkpoints/checkpoint-N/   ← LoRA adapter weights only
  <output>/config.json                  ← final merged GPT2LMHeadModel
  <output>/model.safetensors            ← (compatible with step3_contrastive.py)
  <output>/tokenizer.*

Prerequisites:
  uv add peft

Usage:
  uv run python src/step2_mntp_lora.py \\
      --model  ./bidir_dnagpt \\
      --fasta  ./data/hg38.fa \\
      --output ./mntp_dnagpt_lora

  # Quick smoke-test (no genome file needed):
  uv run python src/step2_mntp_lora.py --model ./bidir_dnagpt --output ./mntp_dnagpt_lora --smoke-test
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
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import MaskedLMOutput

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_fasta_dataset, load_hub_dataset
from step1_bidirectional import patch_to_bidirectional


# ── LoRA configuration ────────────────────────────────────────────────────────

def build_lora_config(args):
    """
    Build LoRA config for GPT-2 (DNAGPT backbone).

    Target modules:
      - c_attn : combined Q/K/V projection in each attention layer (12 total)
      - c_proj  : output projection in attention + MLP (24 total)

    These cover the main weight matrices following LLM2Vec's practice of
    adapting attention projections. Including MLP c_proj adds expressiveness
    at minimal extra cost (~2M additional trainable params for r=16).
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=True,                   # required for GPT-2 Conv1D layers
    )


# ── Model wrapper: GPT-2 + LoRA → LLM2Vec-style MNTP ─────────────────────────

class DNAGPTForMNTPLoRA(nn.Module):
    """
    LoRA-adapted bidirectional GPT-2 for Masked Next Token Prediction.

    Architecture:
      PeftModelForCausalLM (LoRA adapters on c_attn / c_proj)
        └── GPT2LMHeadModel (frozen base weights + patched bidirectional attn)
              ├── transformer (GPT2Model)
              └── lm_head (tied to transformer.wte)

    Loss (LLM2Vec-style, shifted):
      logits[:, :-1] paired with labels[:, 1:]
      → at each position i-1, predict the original token at masked position i.

    Saving:
      Intermediate checkpoints → LoRA adapter weights only (small, fast).
      Final output             → LoRA merged into base model → GPT2LMHeadModel
                                 (directly loadable by step3_contrastive.py).
    """

    _keys_to_ignore_on_save = None   # expected by Trainer._issue_warnings_after_load

    def __init__(self, base_model: AutoModelForCausalLM, lora_config):
        super().__init__()
        from peft import get_peft_model
        self.peft_model = get_peft_model(base_model, lora_config)
        self.config     = base_model.config
        # Set by main() after construction; needed by _load_best_model
        self._base_model_path: Optional[str] = None

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        """
        Args:
            input_ids      : (B, T) — masked input (EOS replaces masked tokens)
            attention_mask : (B, T) — 1 real, 0 padding
            labels         : (B, T) — original ids at masked positions, -100 elsewhere

        The peft_model forward pass routes through LoRA-adapted layers and
        the all-ones attention bias (bidirectional patch from Step 1).
        """
        # Full LoRA-adapted forward (bidirectional attention preserved via bias buffer)
        out    = self.peft_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits   # (B, T, V)

        loss = None
        if labels is not None:
            # LLM2Vec shift: predict token at position i from hidden state at i-1
            # Drop the last logit (no target) and the first label (no predictor)
            logits_s = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            labels_s = labels[:, 1:].contiguous()        # (B, T-1)
            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels_s.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_dir: str, merge: bool = True):
        """
        Save model to disk.

        merge=True  (final output):
            Merge LoRA weights into base model and save as a standard
            GPT2LMHeadModel. Directly loadable by step3_contrastive.py.

        merge=False (checkpoints):
            Save only LoRA adapter weights (adapter_model.safetensors +
            adapter_config.json). Much smaller than the full model.
        """
        os.makedirs(save_dir, exist_ok=True)
        if merge:
            merged = self.peft_model.merge_and_unload()
            merged.save_pretrained(save_dir)
        else:
            self.peft_model.save_pretrained(save_dir)
            # Save base config so the checkpoint dir is self-contained
            self.peft_model.base_model.model.config.save_pretrained(save_dir)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.peft_model.enable_input_require_grads()
        self.peft_model.base_model.model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs or {}
        )

    def gradient_checkpointing_disable(self):
        self.peft_model.base_model.model.transformer.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.peft_model.print_trainable_parameters()


# ── Smoke-test dataset ────────────────────────────────────────────────────────

class SyntheticDNADataset(TorchDataset):
    """Tiny synthetic dataset for quick smoke-tests. No real genome needed."""

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
                return_special_tokens_mask=True,
            )
            self.samples.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Throughput callback ───────────────────────────────────────────────────────

class ThroughputCallback(TrainerCallback):
    """Logs tokens/sec, samples/sec, and train/eval perplexity to W&B."""

    def __init__(self, max_length: int):
        self.max_length  = max_length
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        extra = {}

        if "loss" in logs:
            # logs["loss"] is already normalized by gradient_accumulation_steps
            try:
                extra["train_perplexity"] = round(math.exp(min(logs["loss"], 20)), 4)
            except OverflowError:
                extra["train_perplexity"] = float("inf")

        if "eval_loss" in logs:
            try:
                extra["eval_perplexity"] = round(math.exp(min(logs["eval_loss"], 20)), 4)
            except OverflowError:
                extra["eval_perplexity"] = float("inf")

        if self._step_start is not None:
            elapsed = time.perf_counter() - self._step_start
            if elapsed > 0:
                batch_tokens = args.per_device_train_batch_size * self.max_length
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

class MNTPTrainerLoRA(Trainer):
    """
    Trainer for DNAGPTForMNTPLoRA.

    _save            : saves LoRA adapter weights only at each checkpoint.
    _load_best_model : hot-swaps the adapter weights from the best checkpoint
                       without reloading the (frozen) base model.
    """

    def _save(self, output_dir: str, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, merge=False)   # adapter weights only

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        ckpt = self.state.best_model_checkpoint

        try:
            from peft import set_peft_model_state_dict, load_peft_weights
            adapter_weights = load_peft_weights(ckpt)
            set_peft_model_state_dict(self.model.peft_model, adapter_weights)
            print(f"  Loaded best LoRA adapter from: {ckpt}")
        except Exception as e:
            # Fallback: full reload (slower but safe)
            print(f"  [warn] load_peft_weights failed ({e}), falling back to full reload")
            from peft import PeftModel
            dtype  = next(self.model.parameters()).dtype
            device = next(self.model.parameters()).device
            base = AutoModelForCausalLM.from_pretrained(
                self.model._base_model_path,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
            base = patch_to_bidirectional(base)
            self.model.peft_model = PeftModel.from_pretrained(base, ckpt).to(device)
            print(f"  Loaded best LoRA checkpoint from: {ckpt}")


# ── Training arguments ────────────────────────────────────────────────────────

def build_training_args(output_dir: str, args: argparse.Namespace) -> TrainingArguments:
    """
    TrainingArguments tuned for RTX 5090 (Blackwell sm_120) with LoRA.

    Learning rate note:
      LoRA updates only adapter weights (~1% of params). A higher LR than
      full fine-tuning (1e-4 vs 1e-5) is standard practice — the frozen
      base model weights prevent instability from large adapter updates.
    """
    return TrainingArguments(
        output_dir=output_dir,

        # Precision
        bf16=True,
        bf16_full_eval=True,

        # Batch size — LoRA uses less VRAM (frozen layers skip gradient storage),
        # so per_device batch size can be larger than in step2_mntp.py
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Optimiser — higher LR is appropriate for LoRA adapter-only updates
        learning_rate=args.lr,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,

        # Schedule
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",

        # Logging & saving
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Misc
        seed=args.seed,
        run_name=args.run_name or (
            f"step2_lora_mntp_s{args.seed}"
            + (f"_r{args.repeat_index}" if args.repeat_index is not None else "")
        ),
        report_to="wandb",
        torch_compile=True,
        remove_unused_columns=False,
        gradient_checkpointing=args.grad_ckpt,
        prediction_loss_only=True,
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DNA-LLM2Vec Step 2 (LoRA): MNTP")

    # Model
    p.add_argument("--model",  default="./bidir_dnagpt",
                   help="Path to bidirectional model from Step 1")
    p.add_argument("--output", default="./mntp_dnagpt_lora",
                   help="Output directory for the MNTP-fine-tuned model")

    # Data source
    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta",      default=None,
                      help="Path to genome FASTA file (e.g. hg38.fa or hg38.fa.gz)")
    data.add_argument("--dataset",    default=None,
                      help="HuggingFace dataset name")
    data.add_argument("--smoke-test", action="store_true",
                      help="Quick smoke-test with synthetic data (no genome needed)")

    # Data options
    p.add_argument("--max-length",      type=int,   default=1024,
                   help="Max token length per sample (default: 1024)")
    p.add_argument("--stride",          type=int,   default=0,
                   help="Sliding window stride in bp. 0 = non-overlapping (default)")
    p.add_argument("--filter-n",        action="store_true", default=False,
                   help="Drop FASTA chunks containing any N or ambiguous base "
                        "(DNABERT-2 style). Without this flag, non-ACGT characters "
                        "are stripped in-place, creating artificial junctions.")
    p.add_argument("--mlm-probability", type=float, default=0.15,
                   help="Fraction of tokens to mask (default: 0.15)")

    # LoRA hyperparameters
    p.add_argument("--lora-r",       type=int,   default=16,
                   help="LoRA rank (default: 16, same as LLM2Vec)")
    p.add_argument("--lora-alpha",   type=int,   default=32,
                   help="LoRA alpha — effective scale = alpha/r (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="Dropout within LoRA adapters (default: 0.05)")

    # Training
    p.add_argument("--epochs",       type=int,   default=3,    help="Training epochs (default: 3)")
    p.add_argument("--max-steps",    type=int,   default=-1,
                   help="Maximum optimizer steps. -1 disables the limit and uses --epochs only.")
    p.add_argument("--batch-size",   type=int,   default=32,
                   help="Per-device batch size (default: 32; LoRA allows larger batches)")
    p.add_argument("--grad-accum",   type=int,   default=1,    help="Gradient accumulation steps (default: 1)")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate (default: 1e-4; higher than full FT is standard for LoRA)")
    p.add_argument("--warmup-steps", type=int,   default=500,  help="LR warmup steps (default: 500)")
    p.add_argument("--seed",         type=int,   default=42,   help="Random seed (default: 42)")
    p.add_argument("--run-name",     default=None,
                   help="Optional W&B run name. If omitted, a descriptive name is generated.")
    p.add_argument("--repeat-index", type=int, default=None,
                   help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    p.add_argument("--grad-ckpt",    action="store_true",
                   help="Enable gradient checkpointing (~30%% slower, ~60%% less activation VRAM)")
    p.add_argument("--resume",       default=None, metavar="CHECKPOINT_DIR",
                   help="Resume from a checkpoint dir. Pass 'latest' to auto-detect.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 62)
    print("DNA-LLM2Vec  |  Step 2 (LoRA): MNTP Fine-tuning")
    print("=" * 62)
    if device == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Model : {args.model}")
    print(f"  Output: {args.output}")
    print(f"  LoRA  : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Pred  : LLM2Vec i-1 shift (logits[:,:-1] → labels[:,1:])")
    print("=" * 62)

    # ── 1. Tokenizer ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use EOS token as mask token — avoids embedding resize (which breaks
    # weight tying between wte and lm_head when applying LoRA).
    # Analogous to LLM2Vec using underscore '_' for LLaMA/Mistral.
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.eos_token
        print(f"  Mask token : reusing EOS '{tokenizer.eos_token}' (no embedding resize)")
    else:
        print(f"  Mask token : {tokenizer.mask_token}")

    # ── 2. Load bidirectional model ───────────────────────────────────────────
    print("\n[2/5] Loading bidirectional model + applying LoRA")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",   # keeps bias buffer patchable (Step 1 requirement)
    )
    base_model = patch_to_bidirectional(base_model)

    # Build LoRA config and wrap model
    lora_config = build_lora_config(args)
    model = DNAGPTForMNTPLoRA(base_model, lora_config)
    model._base_model_path = args.model   # stored for _load_best_model fallback

    model = model.to(device)
    model.print_trainable_parameters()

    # Gradient checkpointing with LoRA requires input gradients to be enabled
    if args.grad_ckpt:
        model.peft_model.enable_input_require_grads()
        print("  Gradient checkpointing: enabled")

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    print("\n[3/5] Preparing dataset")

    if args.smoke_test:
        print("  Mode: synthetic smoke-test data")
        train_dataset = SyntheticDNADataset(tokenizer, n_samples=256, seq_len=args.max_length)
        eval_dataset  = SyntheticDNADataset(tokenizer, n_samples=64,  seq_len=args.max_length)

    elif args.fasta:
        print(f"  Mode: FASTA — {args.fasta}")
        ds = load_fasta_dataset(
            args.fasta, tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            filter_n=args.filter_n,
        )
        train_dataset = ds["train"]
        eval_dataset  = ds["validation"]

    elif args.dataset:
        print(f"  Mode: HuggingFace dataset — {args.dataset}")
        ds = load_hub_dataset(args.dataset, tokenizer, max_length=args.max_length)
        train_dataset = ds["train"]
        eval_dataset  = ds.get("validation", None)

    else:
        print("  No data source specified. Use --fasta, --dataset, or --smoke-test.")
        sys.exit(1)

    print(f"  Train : {len(train_dataset):,} samples")
    if eval_dataset:
        print(f"  Val   : {len(eval_dataset):,} samples")

    # ── 4. Collator & Trainer ─────────────────────────────────────────────────
    print("\n[4/5] Setting up training")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = build_training_args(
        output_dir=os.path.join(args.output, "checkpoints"),
        args=args,
    )

    trainer = MNTPTrainerLoRA(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[ThroughputCallback(max_length=args.max_length)],
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Training")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size} × {args.grad_accum} accum")
    print(f"  Effective batch : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  MLM probability : {args.mlm_probability}")
    print()

    # Resolve 'latest' resume path
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

    # W&B
    os.environ["WANDB_PROJECT"] = "dna_foundation"
    os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"

    train_result = trainer.train(resume_from_checkpoint=resume)

    # ── Save — merge LoRA into base model for step3 compatibility ─────────────
    print(f"\nMerging LoRA and saving to: {args.output}")
    model.save_pretrained(args.output, merge=True)
    tokenizer.save_pretrained(args.output)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nStep 2 (LoRA) complete.")
    print(f"  Train loss : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Trainable  : LoRA adapters merged into base model at {args.output}")
    print(f"Next step    : python src/step3_contrastive.py --model {args.output}")


if __name__ == "__main__":
    main()
