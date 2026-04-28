"""
DNA-LLM2Vec  |  Step 2 (LoRA, [MASK]-row-only variant)
======================================================
A drop-in alternative to step2_mntp_lora.py that updates ONLY the row of the
token embedding table corresponding to the newly added [MASK] token.
All 50,257 original DNA-BPE embeddings stay frozen.

Why this exists
---------------
LLM2Vec keeps every original token embedding frozen by reusing an existing
in-vocabulary token (underscore for LLaMA / Mistral) as the mask placeholder.
For DNAGPT the only natural reuse candidate is EOS, but EOS carries decoder-
specific semantics (sequence boundary) that the bidirectional encoder still
needs intact. We therefore introduce a dedicated [MASK] token, but to stay
faithful to LLM2Vec's "preserve all pretrained token semantics" principle,
we limit the embedding-table update to a single row: the new [MASK] row.

This produces a cleaner ablation than step2_mntp_lora.py with full
modules_to_save=["wte"]:
  - LoRA-on-attention is the only mechanism that adapts pretrained weights.
  - The [MASK] row still escapes the cold-start problem (it is trained
    to a useful embedding rather than left at the resize-init value).
  - All 50,257 original BPE rows are bitwise identical before and after
    Step 2, so any downstream gain is unambiguously attributable to
    LoRA + [MASK] embedding learning, not to global wte adaptation.

How it differs from step2_mntp_lora.py
--------------------------------------
1. modules_to_save=["wte"] is kept so PEFT serialises the trainable wte
   copy alongside the adapter (otherwise the [MASK] row update would be
   lost at every checkpoint).
2. A backward hook on the trainable wte copy zeros every row except
   mask_token_id, so optimiser steps move only that one row.
3. wte parameters are placed in a no-weight-decay group, because AdamW's
   decoupled weight decay term acts independently of the gradient and
   would otherwise pull every frozen row slowly toward zero.
4. A post-training sanity check verifies that the trainable wte copy
   differs from the frozen original_module ONLY at mask_token_id.

Usage
-----
    uv run python src/step2_mntp_lora_mask_only.py \\
        --model  ./bidir_dnagpt \\
        --fasta  ./data/hg38.fa \\
        --output ./mntp_dnagpt_lora_mask_only

    # Smoke test:
    uv run python src/step2_mntp_lora_mask_only.py \\
        --model ./bidir_dnagpt --output ./test_out --smoke-test
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    set_seed,
)

# ── Reuse heavy lifting from the main Step 2 script ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import ensure_mask_token, load_fasta_dataset, load_hub_dataset  # noqa: E402
from step1_bidirectional import patch_to_bidirectional  # noqa: E402
from step2_mntp_lora import (  # noqa: E402
    DNAGPTForMNTPLoRA,
    FullModelCheckpointCallback,
    SyntheticDNADataset,
    ThroughputCallback,
    _load_base_for_peft_reload,
    build_lora_config,
    build_training_args,
)


# ── Helpers specific to the mask-row-only variant ────────────────────────────


def _get_trainable_wte(peft_model) -> nn.Embedding:
    """
    Return the trainable wte module created by PEFT's modules_to_save wrapper.

    PEFT's ModulesToSaveWrapper exposes:
      - wrapper.original_module       (frozen reference; never updated)
      - wrapper.modules_to_save[name] (trainable deepcopy used at forward time)

    We need the trainable copy so we can register a hook on its weight.
    """
    wte_wrapper = peft_model.base_model.model.transformer.wte

    if not hasattr(wte_wrapper, "modules_to_save"):
        raise RuntimeError(
            "transformer.wte is not wrapped by PEFT's ModulesToSaveWrapper. "
            "Make sure LoraConfig was built with modules_to_save=['wte']."
        )

    active = getattr(wte_wrapper, "_active_adapter", "default")
    if isinstance(active, (list, tuple, set)):
        active = next(iter(active))
    trainable = wte_wrapper.modules_to_save[active]
    return trainable


def restrict_wte_grad_to_mask_row(peft_model, mask_token_id: int) -> None:
    """
    Register a backward hook on the trainable wte copy so that only
    grad[mask_token_id] survives. Every other row receives a zero gradient.

    The hook is attached to the leaf Parameter (wte.weight), so it fires once
    per backward pass regardless of gradient checkpointing or torch.compile.
    """
    if mask_token_id is None:
        raise ValueError("mask_token_id must be set before registering the hook.")

    trainable_wte = _get_trainable_wte(peft_model)
    weight = trainable_wte.weight

    # Sanity: the trainable copy must require grad. (PEFT marks it trainable
    # automatically because it lives outside the LoRA-frozen base.)
    if not weight.requires_grad:
        raise RuntimeError("Trainable wte copy is unexpectedly frozen.")

    def _zero_non_mask_rows(grad: torch.Tensor) -> torch.Tensor:
        new_grad = torch.zeros_like(grad)
        new_grad[mask_token_id] = grad[mask_token_id]
        return new_grad

    weight.register_hook(_zero_non_mask_rows)
    print(
        f"  Registered grad hook on trainable wte: only row "
        f"{mask_token_id} (= [MASK]) will be updated."
    )


# ── Custom Trainer: wte goes into a no-weight-decay group ────────────────────


class MaskRowMNTPTrainer(Trainer):
    """
    Trainer for DNAGPTForMNTPLoRA that keeps the embedding table out of the
    weight-decay group.

    Why: AdamW applies weight decay multiplicatively to the parameter itself,
    independent of the gradient. Even though our hook zeros the gradient of
    every wte row except [MASK], a non-zero weight_decay would still drag all
    frozen rows toward zero at every step:
        w  -=  lr * weight_decay * w
    Putting wte in weight_decay=0 disables this drift.
    """

    def __init__(self, *args, mask_token_id: Optional[int] = None, **kwargs):
        self._mask_token_id = mask_token_id
        super().__init__(*args, **kwargs)

    # ── Save / load behaviour: same as MNTPTrainerLoRA ────────────────────
    def _save(self, output_dir: str, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, merge=False)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        ckpt = self.state.best_model_checkpoint
        try:
            from peft import load_peft_weights, set_peft_model_state_dict
            adapter_weights = load_peft_weights(ckpt)
            set_peft_model_state_dict(self.model.peft_model, adapter_weights)
            print(f"  Loaded best LoRA adapter from: {ckpt}")
        except Exception as e:
            print(f"  [warn] load_peft_weights failed ({e}), falling back to full reload")
            from peft import PeftModel
            dtype = next(self.model.parameters()).dtype
            device = next(self.model.parameters()).device
            tokenizer = getattr(self, "processing_class", None)
            base = _load_base_for_peft_reload(
                self.model._base_model_path, tokenizer, dtype
            )
            self.model.peft_model = PeftModel.from_pretrained(base, ckpt).to(device)
            print(f"  Loaded best LoRA checkpoint from: {ckpt}")

    # ── Optimiser: 3 param groups (decay / no-decay / wte no-decay) ───────
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_"]

        decay_params, no_decay_params, wte_params = [], [], []
        wte_param_names = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # The trainable wte copy lives at
            #   peft_model.base_model.model.transformer.wte.modules_to_save.<adapter>.weight
            # so checking ".wte." in the name is the most robust filter.
            if ".wte." in name or name.endswith(".wte.weight"):
                wte_params.append(p)
                wte_param_names.append(name)
                continue
            if any(nd in name for nd in no_decay_keywords):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        if not wte_params:
            raise RuntimeError(
                "create_optimizer(): no trainable wte parameter found. "
                "Did modules_to_save=['wte'] take effect?"
            )

        groups = [
            {"params": decay_params,    "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": wte_params,      "weight_decay": 0.0},
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in no_decay_params)
        n_wte = sum(p.numel() for p in wte_params)
        print(
            "  Optimiser groups:\n"
            f"    decay        : {n_decay:>12,} params  (wd={self.args.weight_decay})\n"
            f"    no-decay     : {n_nodecay:>12,} params  (wd=0.0)\n"
            f"    wte (no-wd)  : {n_wte:>12,} params  (wd=0.0; only [MASK] row updated via hook)"
        )
        for n in wte_param_names:
            print(f"        wte param matched: {n}")
        return self.optimizer


# ── Sanity helpers ────────────────────────────────────────────────────────────


def snapshot_wte_state(peft_model) -> dict:
    """
    Capture (frozen original wte, trainable wte copy) at a point in time.
    Returns float32 CPU tensors so we can diff them post-training without
    holding extra GPU memory.
    """
    wte_wrapper = peft_model.base_model.model.transformer.wte
    original = wte_wrapper.original_module.weight.detach().float().cpu().clone()
    trainable = _get_trainable_wte(peft_model).weight.detach().float().cpu().clone()
    return {"original": original, "trainable": trainable}


def report_wte_diff(snapshot_before: dict, peft_model, mask_token_id: int) -> None:
    """
    Verify that ONLY the [MASK] row of the trainable wte copy moved during
    training. All other rows must remain bitwise identical to the snapshot.
    """
    after_trainable = _get_trainable_wte(peft_model).weight.detach().float().cpu()
    before_trainable = snapshot_before["trainable"]
    original = snapshot_before["original"]

    # 1. Mask row diff (expected: > 0)
    mask_l2 = float(torch.norm(after_trainable[mask_token_id] - before_trainable[mask_token_id]))
    mask_max = float((after_trainable[mask_token_id] - before_trainable[mask_token_id]).abs().max())

    # 2. Other-row drift (expected: exactly 0 if hook + no-decay both worked)
    delta_full = after_trainable - before_trainable
    other_mask = torch.ones(after_trainable.size(0), dtype=torch.bool)
    other_mask[mask_token_id] = False
    other_l2 = float(torch.norm(delta_full[other_mask]))
    other_max = float(delta_full[other_mask].abs().max())

    # 3. Original module must be unchanged (PEFT freezes it; this is a control)
    orig_after = peft_model.base_model.model.transformer.wte.original_module.weight.detach().float().cpu()
    orig_drift = float((orig_after - original).abs().max())

    print("\n[MASK]-row update sanity check")
    print(f"  [MASK] row L2 delta            : {mask_l2:.6f}   (expected > 0)")
    print(f"  [MASK] row max abs delta       : {mask_max:.6f}")
    print(f"  Other rows L2 delta (sum)      : {other_l2:.6e}  (expected == 0.0)")
    print(f"  Other rows max abs delta       : {other_max:.6e}  (expected == 0.0)")
    print(f"  original_module drift          : {orig_drift:.6e}  (expected == 0.0)")

    if mask_l2 == 0.0:
        print("  WARNING: [MASK] row did not move. Check that the row was reachable "
              "through the training data (collator must actually emit mask_token_id).")
    if other_max > 0.0:
        print("  WARNING: non-mask rows moved. The hook or weight-decay group did not "
              "fully isolate the [MASK] row.")


# ── Argument parsing (delegated to step2_mntp_lora) ──────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DNA-LLM2Vec Step 2 (LoRA, [MASK]-row-only variant)"
    )

    p.add_argument("--model",  default="./bidir_dnagpt",
                   help="Path to bidirectional model from Step 1")
    p.add_argument("--output", default="./mntp_dnagpt_lora_mask_only",
                   help="Output directory")

    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta",      default=None)
    data.add_argument("--dataset",    default=None)
    data.add_argument("--smoke-test", action="store_true")

    p.add_argument("--max-length",      type=int,   default=1024)
    p.add_argument("--stride",          type=int,   default=0)
    p.add_argument("--filter-n",        action="store_true", default=False)
    p.add_argument("--mlm-probability", type=float, default=0.15)

    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--max-steps",    type=int,   default=-1)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--grad-accum",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int,   default=500)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--run-name",     default=None)
    p.add_argument("--repeat-index", type=int,   default=None)
    p.add_argument("--grad-ckpt",    action="store_true")
    p.add_argument("--full-save-steps", type=int, default=1000)
    p.add_argument("--resume",       default=None, metavar="CHECKPOINT_DIR")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print("DNA-LLM2Vec  |  Step 2 (LoRA, [MASK]-row-only)")
    print("=" * 64)
    if device == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model : {args.model}")
    print(f"  Output: {args.output}")
    print(f"  LoRA  : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Pred  : LLM2Vec i-1 shift (logits[:,:-1] labels[:,1:])")
    print(f"  Embed : ONLY [MASK] row trained; all other token embeddings frozen")
    print("=" * 64)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load bidir base + add [MASK] + apply LoRA ──────────────────────
    print("\n[2/5] Loading bidirectional model + applying LoRA")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    base_model = patch_to_bidirectional(base_model)

    added_mask = ensure_mask_token(tokenizer, base_model)
    if added_mask:
        print(f"  Mask token : added dedicated {tokenizer.mask_token!r}")
    else:
        print(f"  Mask token : {tokenizer.mask_token}")

    # Sanity: [MASK] must be distinct from EOS/PAD so masked positions are not
    # conflated with decoder boundary/padding semantics. PAD is still allowed
    # to share the EOS id, matching the existing DNAGPT setup.
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    if pad_id is None or eos_id is None or mask_id is None:
        raise RuntimeError(
            f"Tokenizer special token ids missing: pad={pad_id}, eos={eos_id}, mask={mask_id}"
        )
    if mask_id in {pad_id, eos_id}:
        raise RuntimeError(
            f"[MASK] token id must differ from pad/eos, got pad={pad_id}, eos={eos_id}, mask={mask_id}."
        )

    lora_config = build_lora_config(args)
    model = DNAGPTForMNTPLoRA(base_model, lora_config)
    model._base_model_path = args.model
    model = model.to(device)
    model.print_trainable_parameters()

    mask_token_id = tokenizer.mask_token_id
    print(f"  Mask token id : {mask_token_id}")

    # Restrict embedding-table updates to the [MASK] row only.
    restrict_wte_grad_to_mask_row(model.peft_model, mask_token_id)

    # Snapshot wte BEFORE training for the post-hoc sanity check.
    wte_snapshot = snapshot_wte_state(model.peft_model)

    if args.grad_ckpt:
        model.peft_model.enable_input_require_grads()
        print("  Gradient checkpointing: enabled")

    # ── 3. Dataset ────────────────────────────────────────────────────────
    print("\n[3/5] Preparing dataset")

    if args.smoke_test:
        print("  Mode: synthetic smoke-test data")
        train_dataset = SyntheticDNADataset(tokenizer, n_samples=256, seq_len=args.max_length)
        eval_dataset  = SyntheticDNADataset(tokenizer, n_samples=64,  seq_len=args.max_length)
    elif args.fasta:
        print(f"  Mode: FASTA {args.fasta}")
        ds = load_fasta_dataset(
            args.fasta, tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            filter_n=args.filter_n,
        )
        train_dataset = ds["train"]
        eval_dataset  = ds["validation"]
    elif args.dataset:
        print(f"  Mode: HuggingFace dataset {args.dataset}")
        ds = load_hub_dataset(args.dataset, tokenizer, max_length=args.max_length)
        train_dataset = ds["train"]
        eval_dataset  = ds.get("validation", None)
    else:
        print("  No data source specified. Use --fasta, --dataset, or --smoke-test.")
        sys.exit(1)

    print(f"  Train : {len(train_dataset):,} samples")
    if eval_dataset:
        print(f"  Val   : {len(eval_dataset):,} samples")

    # ── 4. Trainer ────────────────────────────────────────────────────────
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

    callbacks = [ThroughputCallback(max_length=args.max_length)]
    if args.max_steps > 0 and args.full_save_steps > 0:
        callbacks.append(
            FullModelCheckpointCallback(
                output_root=args.output,
                tokenizer=tokenizer,
                every_steps=args.full_save_steps,
            )
        )

    trainer = MaskRowMNTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        mask_token_id=mask_token_id,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\n[5/5] Training")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Max steps       : {args.max_steps}")
    print(f"  Batch size      : {args.batch_size} × {args.grad_accum} accum")
    print(f"  Effective batch : {args.batch_size * args.grad_accum}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  MLM probability : {args.mlm_probability}")
    vocab_size = model.peft_model.base_model.model.get_input_embeddings().weight.shape[0]
    print(f"  Trainable wte   : 1 row of {vocab_size}")
    print()

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
        print(f"  Resuming from : {resume or 'none found  starting fresh'}")

    os.environ["WANDB_PROJECT"] = "dna_foundation"
    os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"

    train_result = trainer.train(resume_from_checkpoint=resume)

    # ── Post-training sanity check (only [MASK] row should have moved) ────
    report_wte_diff(wte_snapshot, model.peft_model, mask_token_id)

    # ── Save merged model ─────────────────────────────────────────────────
    print(f"\nMerging LoRA and saving to: {args.output}")
    model.save_pretrained(args.output, merge=True)
    tokenizer.save_pretrained(args.output)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\nStep 2 ([MASK]-row-only) complete.")
    print(f"  Train loss : {metrics.get('train_loss', 'N/A')}")
    print(f"  Output     : {args.output}")
    print(f"Next step    : python src/step3_contrastive_lora.py --model {args.output}")


if __name__ == "__main__":
    main()
