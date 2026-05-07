"""
Step 2 for the HyenaDNA branch: span-masked MNTP adaptation.

H0: pretrained causal HyenaDNA baseline
H1: pretrained weights + bidirectional Hyena receptive-field patch
H2: H1 adapted with span-masked next-token prediction (LLM2Vec-style shift)
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, set_seed
from transformers.modeling_outputs import MaskedLMOutput

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_utils import VALID_BASES, _iter_fasta, ensure_mask_token  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_HYENA_MODEL,
    encode_sequence,
    load_hyena_causal_lm,
    load_hyena_tokenizer,
    save_hyena_checkpoint,
)
from hyena_bidirectional import inspect_hyenadna_bidirectional, make_hyenadna_bidirectional  # noqa: E402
from hyena_masking import HyenaDNASpanMaskingCollator, SpanMaskingConfig  # noqa: E402


class SyntheticDNADataset(TorchDataset):
    BASES = ["A", "C", "G", "T"]

    def __init__(self, tokenizer, n_samples: int = 64, seq_len: int = 128):
        self.samples = []
        for _ in range(n_samples):
            seq = "".join(random.choices(self.BASES, k=seq_len))
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


class HyenaDNAForMaskedAdaptation(nn.Module):
    """
    Wrap HyenaDNA for span-masked MNTP.

    The masking policy may hide single positions or contiguous spans, but the
    supervision follows the LLM2Vec-style shifted objective:
      logits[:, i-1] predict labels[:, i]

    This keeps the prediction interface aligned with causal pretraining while
    using bidirectional context to recover masked characters/spans.
    """

    _keys_to_ignore_on_save = None

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits

        loss = None
        if labels is not None:
            # LLM2Vec-style MNTP shift: use position i-1 to predict the masked
            # token at position i. labels are -100 at non-masked positions.
            logits_s = logits[:, :-1, :].contiguous()
            labels_s = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels_s.view(-1),
                ignore_index=-100,
            )

        return MaskedLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_dir: str):
        save_hyena_checkpoint(self.base_model, None, save_dir)


class HyenaTrainer(Trainer):
    """
    Trainer variant that avoids the default shared-tensor safetensors path.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def build_training_args(args) -> TrainingArguments:
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    eval_strategy = "steps" if args.eval_steps > 0 else "epoch"
    save_strategy = "steps" if args.save_steps > 0 else "epoch"

    kwargs = dict(
        output_dir=os.path.join(args.output, "trainer_state"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        prediction_loss_only=True,
        report_to=args.report_to,
        seed=args.seed,
        run_name=args.run_name,
        bf16=bf16_ok,
        bf16_full_eval=bf16_ok,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_params:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy
    if "data_seed" in training_args_params:
        kwargs["data_seed"] = args.seed
    if "save_safetensors" in training_args_params:
        kwargs["save_safetensors"] = False
    return TrainingArguments(**kwargs)


def _clean_sequence(seq: str) -> str:
    return seq.strip().upper()


def _raw_chunk_length(tokenizer, max_length: int) -> int:
    return max(1, max_length - tokenizer.num_special_tokens_to_add(pair=False))


def load_hyena_fasta_dataset(
    fasta_path: str,
    tokenizer,
    max_length: int,
    stride: int,
    val_fraction: float,
    seed: int,
    filter_n: bool,
) -> DatasetDict:
    chunk_len = _raw_chunk_length(tokenizer, max_length)
    effective_stride = stride if stride > 0 else chunk_len

    chunk_count = [0]
    filtered_count = [0]

    def sequence_generator():
        for contig_seq in _iter_fasta(fasta_path, strip_n=not filter_n):
            for start in range(0, len(contig_seq) - chunk_len + 1, effective_stride):
                chunk = contig_seq[start : start + chunk_len]
                if filter_n and VALID_BASES.search(chunk):
                    filtered_count[0] += 1
                    continue
                chunk_count[0] += 1
                yield {"sequence": chunk}

    raw = Dataset.from_generator(sequence_generator)
    print(f"  Total chunks : {chunk_count[0]:,}")
    if filter_n:
        print(f"  Filtered out : {filtered_count[0]:,} chunks with non-ACGT bases")

    def tokenise(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized = raw.map(
        tokenise,
        batched=True,
        batch_size=512,
        remove_columns=["sequence"],
        desc="Tokenising",
    )
    split = tokenized.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_hyena_text_dataset(
    sequences_file: str,
    tokenizer,
    max_length: int,
    val_fraction: float,
    seed: int,
    filter_n: bool,
) -> DatasetDict:
    sequences = []
    dropped = 0
    with open(sequences_file, "r", encoding="utf-8") as handle:
        for line in handle:
            seq = _clean_sequence(line)
            if not seq:
                continue
            if filter_n and VALID_BASES.search(seq):
                dropped += 1
                continue
            sequences.append({"sequence": seq})

    print(f"  Loaded sequences : {len(sequences):,}")
    if filter_n:
        print(f"  Filtered out     : {dropped:,} sequences with non-ACGT bases")

    raw = Dataset.from_list(sequences)

    def tokenise(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized = raw.map(
        tokenise,
        batched=True,
        batch_size=512,
        remove_columns=["sequence"],
        desc="Tokenising",
    )
    split = tokenized.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def smoke_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("HyenaDNA  |  Step 2 Smoke Test (Span-Masked MNTP)")
    print("=" * 72)

    tokenizer = load_hyena_tokenizer(args.model)
    base_model, load_path = load_hyena_causal_lm(args.model, device=device)
    print(f"  Load path                : {load_path}")

    base_model = make_hyenadna_bidirectional(base_model)
    report = inspect_hyenadna_bidirectional(base_model)
    print(f"  HyenaFilter modules      : {report.total_hyena_filters}")
    print(f"  Modules with bidir=True  : {report.modules_with_bidirectional_true}")
    print(f"  Forward-patched modules  : {report.modules_forward_patched}")

    ensure_mask_token(tokenizer, base_model)
    if tokenizer.mask_token_id in {tokenizer.pad_token_id, tokenizer.eos_token_id}:
        raise RuntimeError("[MASK] token must be distinct from PAD/EOS for HyenaDNA adaptation.")

    if hasattr(base_model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    wrapper = HyenaDNAForMaskedAdaptation(base_model)
    wrapper.train()

    dataset = SyntheticDNADataset(tokenizer, n_samples=8, seq_len=min(args.max_length, 128))
    collator = HyenaDNASpanMaskingCollator(
        tokenizer=tokenizer,
        config=SpanMaskingConfig(
            mask_probability=args.mask_probability,
            masking_mode=args.masking_mode,
            span_min_length=args.span_min_length,
            span_max_length=args.span_max_length,
        ),
    )

    batch = collator([dataset[i] for i in range(4)])
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr)
    out = wrapper(**batch)
    loss = out.loss
    if loss is None:
        raise RuntimeError("Smoke test failed: span-masked MNTP loss is None.")
    print(f"  One-step MNTP loss        : {loss.item():.6f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    wrapper.eval()
    emb = encode_sequence(wrapper.base_model, tokenizer, "ACGT" * 32, pooling="mean", max_length=args.max_length, device=device)
    print(f"  Mean pooled embedding     : {tuple(emb.shape)}")
    print(f"  Embedding L2 norm         : {float(torch.linalg.norm(emb)):.6f}")
    print("  Smoke test               : PASSED")


def parse_args():
    p = argparse.ArgumentParser(description="HyenaDNA Step 2: span-masked MNTP adaptation")
    p.add_argument("--model", default=DEFAULT_HYENA_MODEL, help="H0 or H1 checkpoint / model id")
    p.add_argument("--output", default="./hyena_h2_masked_adapted")

    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta", default=None)
    data.add_argument("--sequences-file", default=None, help="Plain-text file with one DNA sequence per line")
    data.add_argument("--smoke-test", action="store_true")

    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.01)

    p.add_argument("--mask-probability", type=float, default=0.15)
    p.add_argument("--masking-mode", choices=["single", "span"], default="span")
    p.add_argument("--span-min-length", type=int, default=3)
    p.add_argument("--span-max-length", type=int, default=20)

    p.add_argument("--train-mode", choices=["full", "adapter_or_lora"], default="full")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--report-to", default="wandb", choices=["none", "wandb"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="hyena_step2_span_masked_mntp")
    return p.parse_args()


def main():
    args = parse_args()
    if args.train_mode != "full":
        raise NotImplementedError(
            "HyenaDNA Step 2 currently prioritises full fine-tuning. "
            "adapter_or_lora is intentionally left unimplemented for now."
        )

    set_seed(args.seed)

    if args.smoke_test:
        smoke_test(args)
        return

    if not args.fasta and not args.sequences_file:
        raise ValueError("Provide either --fasta, --sequences-file, or --smoke-test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 72)
    print("HyenaDNA  |  Step 2: Span-Masked MNTP Adaptation")
    print("=" * 72)
    print(f"  Input model              : {args.model}")
    print(f"  Output                   : {args.output}")
    print(f"  Device                   : {device}")
    print(f"  Precision                : {dtype}")
    print(f"  Train mode               : {args.train_mode}")
    print(f"  Masking mode             : {args.masking_mode}")
    print(f"  Mask probability         : {args.mask_probability}")
    print(f"  Seed                     : {args.seed}")
    print("  Objective                : span-masked MNTP (LLM2Vec-style shift)")
    print(f"  Dataloader workers       : {args.dataloader_num_workers}")
    print(f"  Report to                : {args.report_to}")
    if args.masking_mode == "span":
        print(f"  Span length range        : {args.span_min_length}-{args.span_max_length}")

    tokenizer = load_hyena_tokenizer(args.model)
    base_model, load_path = load_hyena_causal_lm(args.model, device=device, dtype=dtype)
    print(f"  Load path                : {load_path}")

    base_model = make_hyenadna_bidirectional(base_model)
    report = inspect_hyenadna_bidirectional(base_model)
    print(f"  HyenaFilter modules      : {report.total_hyena_filters}")
    print(f"  Modules with bidir=True  : {report.modules_with_bidirectional_true}")
    print(f"  Forward-patched modules  : {report.modules_forward_patched}")

    added_mask = ensure_mask_token(tokenizer, base_model)
    if added_mask:
        print("  Mask token               : added dedicated [MASK]")
    else:
        print(f"  Mask token               : existing {tokenizer.mask_token!r}")
    print(f"  Mask token id            : {tokenizer.mask_token_id}")

    if hasattr(base_model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing   : enabled")

    wrapper = HyenaDNAForMaskedAdaptation(base_model)

    if args.fasta:
        print("\n[1/4] Loading FASTA dataset")
        datasets = load_hyena_fasta_dataset(
            fasta_path=args.fasta,
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            val_fraction=args.val_fraction,
            seed=args.seed,
            filter_n=args.filter_n,
        )
    else:
        print("\n[1/4] Loading text-sequence dataset")
        datasets = load_hyena_text_dataset(
            sequences_file=args.sequences_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            val_fraction=args.val_fraction,
            seed=args.seed,
            filter_n=args.filter_n,
        )

    print(f"  Train samples            : {len(datasets['train']):,}")
    print(f"  Validation samples       : {len(datasets['validation']):,}")

    collator = HyenaDNASpanMaskingCollator(
        tokenizer=tokenizer,
        config=SpanMaskingConfig(
            mask_probability=args.mask_probability,
            masking_mode=args.masking_mode,
            span_min_length=args.span_min_length,
            span_max_length=args.span_max_length,
        ),
    )

    os.makedirs(args.output, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    training_args = build_training_args(args)

    trainer = HyenaTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\n[2/4] Training span-masked MNTP adaptation")
    trainer.train()

    print("\n[3/4] Saving H2 checkpoint")
    wrapper.base_model.config.hyena_training_stage = "H2"
    wrapper.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"  Saved H2 checkpoint      : {args.output}")

    print("\n[4/4] Extracting one mean-pooled embedding")
    emb = encode_sequence(wrapper.base_model, tokenizer, "ACGT" * 64, pooling="mean", max_length=args.max_length, device=device)
    print(f"  Embedding shape          : {tuple(emb.shape)}")
    print(f"  Embedding L2 norm        : {float(torch.linalg.norm(emb)):.6f}")


if __name__ == "__main__":
    main()
