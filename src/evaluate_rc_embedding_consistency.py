#!/usr/bin/env python
"""
Evaluate reverse-complement consistency of frozen DNA embeddings.

The script embeds each sequence and its reverse complement with the same frozen
model, then reports cosine similarity and L2 distance. It also reports two
anchors: unrelated random sequences and one-base mutants. This makes the RC
score interpretable instead of a floating number without a scale.

This pooled-vector diagnostic measures strand invariance. Per-position
equivariance needs coordinate and channel alignment and should be evaluated
separately for architectures that define an explicit RC channel transform.

Examples:
  uv run python src/evaluate_rc_embedding_consistency.py \
    --models "H2:./hyena_h2_masked_adapted:hyena" \
    --fasta ./data/hg38.fa --window-bp 4096 --max-samples 1000 \
    --output ./eval_results/rc_embedding_hyena_h2

  uv run python src/evaluate_rc_embedding_consistency.py \
    --models "M4:./contrastive_dnagpt_revcomp_lora:bidir" \
    --sequences-file ./data/example_sequences.txt \
    --output ./eval_results/rc_embedding_m4
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
HYENA_DIR = os.path.join(ROOT, "src_hyena")
for path in (ROOT, SRC_DIR, HYENA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import step4_evaluate as base  # noqa: E402
import step4_hyena_evaluate as hyena_eval  # noqa: E402
from data_utils import VALID_BASES, _iter_fasta  # noqa: E402


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


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
                "where MODE is causal, bidir, encoder, or hyena."
            )
        name, path, mode = parts
        if mode not in {"causal", "bidir", "encoder", "hyena"}:
            raise ValueError(f"Unsupported mode {mode!r}.")
        return ModelSpec(name=name, path=path, mode=mode)


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def clean_sequence(seq: str) -> str:
    return seq.strip().upper()


def load_text_sequences(path: str, max_samples: int | None, filter_n: bool) -> list[str]:
    seqs: list[str] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            seq = clean_sequence(line)
            if not seq:
                continue
            if filter_n and VALID_BASES.search(seq):
                dropped += 1
                continue
            seqs.append(seq)
            if max_samples and len(seqs) >= max_samples:
                break
    if filter_n:
        print(f"  Dropped non-ACGT sequences : {dropped:,}")
    return seqs


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
    for contig in _iter_fasta(fasta_path, strip_n=not filter_n):
        for start in range(0, len(contig) - window_bp + 1, stride):
            seq = contig[start : start + window_bp].upper()
            if filter_n and VALID_BASES.search(seq):
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
        valid = [idx for idx, base in enumerate(chars) if base in "ACGT"]
        if not valid:
            mutants.append(seq)
            continue
        idx = rng.choice(valid)
        chars[idx] = rng.choice([base for base in "ACGT" if base != chars[idx]])
        mutants.append("".join(chars))
    return mutants


def load_model(spec: ModelSpec, device: str, dtype: torch.dtype):
    if spec.mode == "hyena":
        hyena_spec = base.ModelSpec(name=spec.name, path=spec.path, mode="encoder")
        model, tokenizer = hyena_eval.load_model(hyena_spec, device=device, dtype=dtype)
        return model, tokenizer, hyena_eval.encode_sequences

    generic_spec = base.ModelSpec(name=spec.name, path=spec.path, mode=spec.mode)
    model, tokenizer = base.load_model(generic_spec, device=device, dtype=dtype)
    return model, tokenizer, base.encode_sequences


def encode(
    encode_fn,
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
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pooling=pooling,
        desc=desc,
    )
    if encode_fn is base.encode_sequences:
        kwargs["normalize"] = normalize
    emb = encode_fn(**kwargs)
    if normalize and encode_fn is not base.encode_sequences:
        denom = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(denom, 1e-12, None)
    return emb.astype(np.float32, copy=False)


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
    return cosine, l2


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RC consistency of frozen DNA embeddings.")
    p.add_argument("--models", nargs="+", required=True, help="NAME:PATH:MODE specs; MODE includes hyena.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequences-file", default=None, help="Plain text file with one DNA sequence per line.")
    src.add_argument("--fasta", default=None, help="FASTA file to window into examples.")
    src.add_argument("--smoke-test", action="store_true")
    p.add_argument("--window-bp", type=int, default=4096)
    p.add_argument("--stride-bp", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--filter-n", action="store_true", help="Drop sequences/windows containing non-ACGT bases.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hyena-batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--hyena-max-length", type=int, default=8192)
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--output", default="./eval_results/rc_embedding_consistency")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        sequences = smoke_sequences(args.max_samples, min(args.window_bp, 512), args.seed)
    elif args.sequences_file:
        sequences = load_text_sequences(args.sequences_file, args.max_samples, args.filter_n)
    else:
        sequences = load_fasta_windows(args.fasta, args.window_bp, args.stride_bp, args.max_samples, args.filter_n)

    if not sequences:
        raise SystemExit("No sequences loaded.")
    rc_sequences = [reverse_complement(seq) for seq in sequences]
    random_sequences = random_dna_like(sequences, args.seed + 1009)
    mutant_sequences = point_mutants(sequences, args.seed + 2003)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.fp32 or device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    specs = [ModelSpec.parse(item) for item in args.models]
    os.makedirs(args.output, exist_ok=True)

    print("RC embedding consistency")
    print(f"  Sequences : {len(sequences):,}")
    print(f"  Device    : {device}")
    print(f"  Pooling   : {args.pooling}")
    print(f"  Output    : {args.output}")

    all_summary = {}
    raw_rows = []
    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer, encode_fn = load_model(spec, device, dtype)
        if spec.mode == "hyena":
            batch_size = args.hyena_batch_size or args.batch_size
        else:
            batch_size = args.batch_size
        max_length = args.hyena_max_length if spec.mode == "hyena" else args.max_length

        fwd = encode(
            encode_fn, model, tokenizer, sequences, batch_size, max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} forward",
        )
        rc = encode(
            encode_fn, model, tokenizer, rc_sequences, batch_size, max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} rc",
        )
        random_emb = encode(
            encode_fn, model, tokenizer, random_sequences, batch_size, max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} random anchor",
        )
        mutant_emb = encode(
            encode_fn, model, tokenizer, mutant_sequences, batch_size, max_length, device,
            args.pooling, normalize=not args.no_l2_normalize, desc=f"{spec.name} mutant anchor",
        )

        fwd_norm = np.linalg.norm(fwd, axis=1)
        rc_norm = np.linalg.norm(rc, axis=1)
        cosine, l2 = pair_metrics(fwd, rc)
        random_cosine, random_l2 = pair_metrics(fwd, random_emb)
        mutant_cosine, mutant_l2 = pair_metrics(fwd, mutant_emb)

        all_summary[spec.name] = {
            "model": spec.__dict__,
            "n": len(sequences),
            "rc_invariance": {
                "cosine": summarize(cosine),
                "l2": summarize(l2),
            },
            "random_anchor": {
                "cosine": summarize(random_cosine),
                "l2": summarize(random_l2),
            },
            "point_mutant_anchor": {
                "cosine": summarize(mutant_cosine),
                "l2": summarize(mutant_l2),
            },
            "forward_norm": summarize(fwd_norm),
            "rc_norm": summarize(rc_norm),
        }
        print(
            f"  RC cosine mean={all_summary[spec.name]['rc_invariance']['cosine']['mean']:.4f} "
            f"median={all_summary[spec.name]['rc_invariance']['cosine']['median']:.4f} "
            f"p05={all_summary[spec.name]['rc_invariance']['cosine']['p05']:.4f}"
        )
        print(
            f"  anchors   random={all_summary[spec.name]['random_anchor']['cosine']['mean']:.4f} "
            f"mutant={all_summary[spec.name]['point_mutant_anchor']['cosine']['mean']:.4f}"
        )

        for idx, seq in enumerate(sequences):
            raw_rows.append({
                "model": spec.name,
                "index": idx,
                "length": len(seq),
                "rc_cosine": float(cosine[idx]),
                "rc_l2": float(l2[idx]),
                "random_cosine": float(random_cosine[idx]),
                "random_l2": float(random_l2[idx]),
                "mutant_cosine": float(mutant_cosine[idx]),
                "mutant_l2": float(mutant_l2[idx]),
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
                "pooling": args.pooling,
                "l2_normalize": not args.no_l2_normalize,
                "metric_scope": "pooled_embedding_invariance",
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
                "length",
                "rc_cosine",
                "rc_l2",
                "random_cosine",
                "random_l2",
                "mutant_cosine",
                "mutant_l2",
                "sequence_prefix",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved raw CSV: {raw_path}")


if __name__ == "__main__":
    main()
