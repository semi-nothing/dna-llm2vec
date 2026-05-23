"""
Evo Step 2: masked next-token prediction repair with LoRA.

Creates E2 from E1 using the same LLM2Vec-style shifted MNTP objective used for
DNAGPT, but always trains through PEFT LoRA adapters and saves a merged Evo
checkpoint for Step 3/4/5 compatibility.
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
from transformers.modeling_outputs import MaskedLMOutput

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    DEFAULT_EVO_MODEL,
    VALID_BASES_RE,
    count_trainable_parameters_m,
    infer_lora_target_modules,
    load_evo_causal_lm,
    load_evo_tokenizer,
    resolve_mask_surrogate_id,
    save_evo_checkpoint,
)


class RawSequenceDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


def load_fasta_windows(path: str, window_bp: int, stride_bp: int, filter_n: bool) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    stride = stride_bp or window_bp

    def flush():
        if not current:
            return
        seq = "".join(current).upper()
        for start in range(0, max(0, len(seq) - window_bp + 1), stride):
            window = seq[start : start + window_bp]
            if len(window) == window_bp and (not filter_n or not VALID_BASES_RE.search(window)):
                sequences.append(window)
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
    return sequences


def smoke_sequences(n: int, seq_len: int) -> list[str]:
    return ["".join(random.choices("ACGT", k=seq_len)) for _ in range(n)]


@dataclass
class EvoMaskingCollator:
    tokenizer: object
    max_length: int
    mlm_probability: float
    mask_token_id: int
    replacement_policy: str
    random_token_ids: torch.Tensor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [item["sequence"].upper() for item in features]
        enc = self.tokenizer(
            seqs,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        probability = torch.full(labels.shape, self.mlm_probability)
        special = torch.zeros_like(labels, dtype=torch.bool)
        for tok_id in (
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        ):
            if tok_id is not None:
                special |= labels.eq(int(tok_id))
        probability.masked_fill_(special, 0.0)
        masked = torch.bernoulli(probability).bool()
        labels[~masked] = -100
        idx = masked.nonzero(as_tuple=True)
        if idx[0].numel() > 0:
            if self.replacement_policy == "all_mask":
                input_ids[idx] = int(self.mask_token_id)
            elif self.replacement_policy == "bert":
                probs = torch.rand(idx[0].numel(), device=input_ids.device)
                mask_positions = probs < 0.8
                random_positions = (probs >= 0.8) & (probs < 0.9)
                if mask_positions.any():
                    input_ids[idx[0][mask_positions], idx[1][mask_positions]] = int(self.mask_token_id)
                if random_positions.any():
                    n_random = int(random_positions.sum().item())
                    rand_idx = torch.randint(
                        0,
                        self.random_token_ids.numel(),
                        (n_random,),
                        device=input_ids.device,
                    )
                    random_ids = self.random_token_ids.to(input_ids.device)
                    input_ids[idx[0][random_positions], idx[1][random_positions]] = random_ids[rand_idx]
            else:
                raise ValueError(f"Unsupported replacement_policy={self.replacement_policy!r}")
        enc["input_ids"] = input_ids
        enc["labels"] = labels
        return enc


def resolve_random_base_token_ids(tokenizer) -> torch.Tensor:
    ids = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for base in ("A", "C", "G", "T"):
        token_ids = tokenizer.encode(base, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Random replacement base {base!r} must encode to one token; got {token_ids}.")
        token_id = int(token_ids[0])
        if unk_id is not None and token_id == int(unk_id):
            raise ValueError(f"Random replacement base {base!r} resolved to unk token id {token_id}.")
        ids.append(token_id)
    if len(set(ids)) != 4:
        raise ValueError(f"Random replacement A/C/G/T ids must be distinct; got {ids}.")
    return torch.tensor(ids, dtype=torch.long)


def build_lora_config(model, args):
    from peft import LoraConfig

    targets = infer_lora_target_modules(model, args.lora_target_modules)
    print(f"  LoRA targets             : {targets}")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        modules_to_save=["embed"] if args.save_embeddings else None,
        lora_dropout=args.lora_dropout,
        bias="none",
    )


class EvoForMNTPLoRA(nn.Module):
    def __init__(self, base_model, lora_config):
        super().__init__()
        from peft import get_peft_model

        self.peft_model = get_peft_model(base_model, lora_config)
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        try:
            out = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError as e:
            message = str(e)
            if "attention_mask" not in message and "use_cache" not in message:
                raise
            out = self.peft_model(input_ids=input_ids)
        logits = out.logits
        loss = None
        if labels is not None:
            logits_s = logits[:, :-1, :].contiguous()
            labels_s = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels_s.view(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, output_dir: str, merge: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        if merge:
            model = self.peft_model.merge_and_unload()
            save_evo_checkpoint(model, None, output_dir)
        else:
            self.peft_model.save_pretrained(output_dir)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.peft_model, "enable_input_require_grads"):
            self.peft_model.enable_input_require_grads()
        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs or {"use_reentrant": False})

    def gradient_checkpointing_disable(self):
        if hasattr(self.peft_model, "gradient_checkpointing_disable"):
            self.peft_model.gradient_checkpointing_disable()


class EvoMNTPTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir, merge=False)
        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if proc is not None:
            proc.save_pretrained(output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evo Step 2: MNTP LoRA")
    p.add_argument("--model", default=DEFAULT_EVO_MODEL)
    p.add_argument("--output", default="./evo_e2_mntp_lora")
    data = p.add_mutually_exclusive_group()
    data.add_argument("--fasta")
    data.add_argument("--sequences-file")
    data.add_argument("--smoke-test", action="store_true")
    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--stride", type=int, default=4096)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--mask-token", default="_", help="Existing single Evo token used as the MNTP mask surrogate.")
    p.add_argument("--mask-token-id", type=int, default=95, help="Expected id for --mask-token; set to -1 to disable the check.")
    p.add_argument("--replacement-policy", choices=("bert", "all_mask"), default="bert")
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
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="evo_step2_mntp_lora")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load_data(args):
    if args.smoke_test:
        seqs = smoke_sequences(256, min(args.max_length, 512))
    elif args.fasta:
        seqs = load_fasta_windows(args.fasta, args.max_length, args.stride, args.filter_n)
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
    n_val = max(1, int(len(seqs) * args.val_fraction))
    return RawSequenceDataset(seqs[n_val:]), RawSequenceDataset(seqs[:n_val])


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = load_evo_tokenizer(args.model)
    model, _ = load_evo_causal_lm(args.model, device=device, dtype=dtype)
    expected_mask_id = None if args.mask_token_id < 0 else args.mask_token_id
    mask_token_id = resolve_mask_surrogate_id(tokenizer, args.mask_token, expected_mask_id)
    print(f"  Mask surrogate           : {args.mask_token!r} -> id {mask_token_id}")
    random_token_ids = resolve_random_base_token_ids(tokenizer)
    print(f"  Replacement policy       : {args.replacement_policy}")
    print(f"  Random replacement ids   : {random_token_ids.tolist()}")
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    wrapper = EvoForMNTPLoRA(model, build_lora_config(model, args)).to(device=device)
    print(f"  Trainable parameters     : {count_trainable_parameters_m(wrapper):.2f}M")

    train_ds, val_ds = load_data(args)
    collator = EvoMaskingCollator(
        tokenizer,
        args.max_length,
        args.mlm_probability,
        mask_token_id,
        args.replacement_policy,
        random_token_ids,
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
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=args.run_name,
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
    trainer = EvoMNTPTrainer(**trainer_kwargs)
    trainer.train()
    wrapper.peft_model.base_model.model.config.evo_training_stage = "E2_mntp_lora"
    wrapper.save_pretrained(args.output, merge=True)
    tokenizer.save_pretrained(args.output)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved merged E2 checkpoint to {args.output}")


if __name__ == "__main__":
    main()
