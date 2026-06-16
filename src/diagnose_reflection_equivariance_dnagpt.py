#!/usr/bin/env python
"""
Pooled reflection diagnostic for DNAGPT variants.

This script tests pure positional reversal, not reverse complement:

    reverse("ACGTTA") = "ATTGCA"

For DNAGPT, token-level reflection equivariance is not a clean primary metric,
because BPE token boundaries can change after reversing the raw DNA string. The
main diagnostic here is therefore pooled embedding similarity:

    cos(f(x), f(reverse(x)))

reported relative to random-sequence and point-mutant anchors. A high
reflection-minus-random margin means the model is closer to being orientation
blind after pooling; a margin near zero means reversal is no more similar than
an unrelated sequence under that embedding geometry.

Example:
  uv run python src/diagnose_reflection_equivariance_dnagpt.py \
    --models \
      "M0:dnagpt/human_gpt2-v1:causal" \
      "M1:./bidir_dnagpt:bidir" \
      "M2:./mntp_dnagpt_lora_fn:bidir" \
      "M3:./contrastive_dnagpt_dropout_lora:bidir" \
      "M4:./contrastive_dnagpt_revcomp_lora:bidir" \
      "M5:./contrastive_dnagpt_crop_lora:bidir" \
    --smoke-test --window-bp 4096 --max-samples 256 \
    --output exp_res/reflection_equivariance_dnagpt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import step4_evaluate as step4  # noqa: E402


VALID_BASES = set("ACGT")


@dataclass
class ModelSpec:
    name: str
    path: str
    mode: str

    @staticmethod
    def parse(spec: str) -> "ModelSpec":
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad model spec {spec!r}; expected NAME:PATH:MODE, "
                "where MODE is causal, bidir, or encoder."
            )
        name, path, mode = parts
        if mode not in {"causal", "bidir", "encoder"}:
            raise ValueError(f"Unsupported mode {mode!r}.")
        return ModelSpec(name=name, path=path, mode=mode)


def clean_sequence(seq: str) -> str:
    return seq.strip().upper()


def reverse_sequence(seq: str) -> str:
    return seq[::-1].upper()


def smoke_sequences(n: int, length: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=length)) for _ in range(n)]


def random_dna_like(sequences: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=len(seq))) for seq in sequences]


def point_mutants(sequences: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    mutants = []
    for seq in sequences:
        chars = list(seq)
        valid = [idx for idx, base in enumerate(chars) if base in VALID_BASES]
        if not valid:
            mutants.append(seq)
            continue
        idx = rng.choice(valid)
        chars[idx] = rng.choice([base for base in "ACGT" if base != chars[idx]])
        mutants.append("".join(chars))
    return mutants


def load_text_sequences(path: str, max_samples: int | None, filter_n: bool) -> list[str]:
    seqs: list[str] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            seq = clean_sequence(line)
            if not seq:
                continue
            if filter_n and set(seq) - VALID_BASES:
                dropped += 1
                continue
            seqs.append(seq)
            if max_samples and len(seqs) >= max_samples:
                break
    if filter_n:
        print(f"  Dropped non-ACGT sequences : {dropped:,}")
    return seqs


def iter_fasta_contigs(path: str):
    cur: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    yield "".join(cur).upper()
                    cur = []
            else:
                cur.append(line)
    if cur:
        yield "".join(cur).upper()


def load_fasta_windows(
    fasta_path: str,
    window_bp: int,
    stride_bp: int,
    max_samples: int | None,
    filter_n: bool,
) -> list[str]:
    stride = stride_bp if stride_bp > 0 else window_bp
    seqs: list[str] = []
    dropped = 0
    for contig in iter_fasta_contigs(fasta_path):
        for start in range(0, len(contig) - window_bp + 1, stride):
            seq = contig[start : start + window_bp].upper()
            if filter_n and set(seq) - VALID_BASES:
                dropped += 1
                continue
            seqs.append(seq)
            if max_samples and len(seqs) >= max_samples:
                if filter_n:
                    print(f"  Dropped non-ACGT windows   : {dropped:,}")
                return seqs
    if filter_n:
        print(f"  Dropped non-ACGT windows   : {dropped:,}")
    return seqs


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def pair_metrics(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.clip(a_norm * b_norm, 1e-12, None)
    l2 = np.linalg.norm(a - b, axis=1)
    return cosine.astype(np.float32), l2.astype(np.float32)


def load_model(spec: ModelSpec, device: str, dtype: torch.dtype):
    step4_spec = step4.ModelSpec(name=spec.name, path=spec.path, mode=spec.mode)
    return step4.load_model(step4_spec, device=device, dtype=dtype)


def encode(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    normalize: bool,
    desc: str,
) -> np.ndarray:
    return step4.encode_sequences(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pooling=pooling,
        normalize=normalize,
        desc=desc,
    ).astype(np.float32, copy=False)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DNAGPT pooled reflection consistency.")
    p.add_argument("--models", nargs="+", required=True, help="NAME:PATH:MODE specs.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequences-file", default=None, help="Plain text file with one DNA sequence per line.")
    src.add_argument("--fasta", default=None, help="FASTA file to window into examples.")
    src.add_argument("--smoke-test", action="store_true", help="Use random ACGT sequences.")
    p.add_argument("--window-bp", type=int, default=4096, help="Sequence length in base pairs.")
    p.add_argument("--stride-bp", type=int, default=0, help="FASTA stride; default equals --window-bp.")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--filter-n", action="store_true", help="Drop sequences/windows containing non-ACGT bases.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length in tokens.")
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--output", default="./exp_res/reflection_equivariance_dnagpt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke_test:
        sequences = smoke_sequences(args.max_samples, args.window_bp, args.seed)
        source = "random-ACGT"
    elif args.sequences_file:
        sequences = load_text_sequences(args.sequences_file, args.max_samples, args.filter_n)
        source = args.sequences_file
    else:
        sequences = load_fasta_windows(args.fasta, args.window_bp, args.stride_bp, args.max_samples, args.filter_n)
        source = args.fasta

    if not sequences:
        raise SystemExit("No sequences loaded.")

    reversed_sequences = [reverse_sequence(seq) for seq in sequences]
    random_sequences = random_dna_like(sequences, args.seed + 1009)
    mutant_sequences = point_mutants(sequences, args.seed + 2003)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.fp32 or device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    specs = [ModelSpec.parse(item) for item in args.models]
    os.makedirs(args.output, exist_ok=True)

    print("DNAGPT pooled reflection diagnostic")
    print(f"  Sequences : {len(sequences):,} ({source})")
    print(f"  Window bp : {args.window_bp:,}")
    print(f"  Max tokens: {args.max_length:,}")
    print(f"  Device    : {device}")
    print(f"  Pooling   : {args.pooling}")
    print("  Reverse   : positional reversal only, no complement")
    print(f"  Output    : {args.output}")

    all_summary = {}
    raw_rows = []
    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer = load_model(spec, device, dtype)

        fwd = encode(
            model, tokenizer, sequences, args.batch_size, args.max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} forward",
        )
        rev = encode(
            model, tokenizer, reversed_sequences, args.batch_size, args.max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} reverse",
        )
        random_emb = encode(
            model, tokenizer, random_sequences, args.batch_size, args.max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} random anchor",
        )
        mutant_emb = encode(
            model, tokenizer, mutant_sequences, args.batch_size, args.max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} mutant anchor",
        )

        reflection_cosine, reflection_l2 = pair_metrics(fwd, rev)
        random_cosine, random_l2 = pair_metrics(fwd, random_emb)
        mutant_cosine, mutant_l2 = pair_metrics(fwd, mutant_emb)
        margin = reflection_cosine - random_cosine

        all_summary[spec.name] = {
            "model": spec.__dict__,
            "n": len(sequences),
            "reflection": {
                "cosine": summarize(reflection_cosine),
                "l2": summarize(reflection_l2),
            },
            "random_anchor": {
                "cosine": summarize(random_cosine),
                "l2": summarize(random_l2),
            },
            "point_mutant_anchor": {
                "cosine": summarize(mutant_cosine),
                "l2": summarize(mutant_l2),
            },
            "reflection_margin_over_random": summarize(margin),
        }
        print(
            f"  reflection cosine mean={all_summary[spec.name]['reflection']['cosine']['mean']:.4f} "
            f"median={all_summary[spec.name]['reflection']['cosine']['median']:.4f} "
            f"p05={all_summary[spec.name]['reflection']['cosine']['p05']:.4f}"
        )
        print(
            f"  anchors            random={all_summary[spec.name]['random_anchor']['cosine']['mean']:.4f} "
            f"mutant={all_summary[spec.name]['point_mutant_anchor']['cosine']['mean']:.4f} "
            f"margin={all_summary[spec.name]['reflection_margin_over_random']['mean']:+.4f}"
        )

        for idx, seq in enumerate(sequences):
            raw_rows.append({
                "model": spec.name,
                "index": idx,
                "length_bp": len(seq),
                "reflection_cosine": float(reflection_cosine[idx]),
                "reflection_l2": float(reflection_l2[idx]),
                "random_cosine": float(random_cosine[idx]),
                "random_l2": float(random_l2[idx]),
                "mutant_cosine": float(mutant_cosine[idx]),
                "mutant_l2": float(mutant_l2[idx]),
                "reflection_margin_over_random": float(margin[idx]),
                "sequence_prefix": seq[:80],
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_sequences": len(sequences),
                "source": source,
                "window_bp": args.window_bp,
                "max_length": args.max_length,
                "pooling": args.pooling,
                "l2_normalize": not args.no_l2_normalize,
                "metric_scope": "pooled_embedding_positional_reversal",
                "reverse_definition": "seq[::-1], no complement",
                "anchors": ["random_unrelated_sequence", "single_point_mutant"],
                "models": all_summary,
            },
            f,
            indent=2,
        )

    raw_path = os.path.join(args.output, "raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "index",
                "length_bp",
                "reflection_cosine",
                "reflection_l2",
                "random_cosine",
                "random_l2",
                "mutant_cosine",
                "mutant_l2",
                "reflection_margin_over_random",
                "sequence_prefix",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved raw CSV: {raw_path}")


if __name__ == "__main__":
    main()
