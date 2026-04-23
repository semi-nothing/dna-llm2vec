"""
Analyze reverse-complement tokenization mismatch for DNA BPE tokenizers.

This script tests the hypothesis that reverse-complement contrastive pairs are
noisy for DNAGPT-style BPE tokenization because reverse-complementing a DNA
sequence can change both token identities and token boundaries.

Examples:
  uv run python src/analyze_revcomp_tokenization.py \
      --model dnagpt/human_gpt2-v1 \
      --fasta ./data/hg38.fa \
      --max-samples 2000 \
      --seq-len 4096 \
      --output ./eval_results/revcomp_tokenization_hg38.json

  uv run python src/analyze_revcomp_tokenization.py \
      --model dnagpt/human_gpt2-v1 \
      --gue-plus-dir ./data/GUE_plus \
      --max-samples 2000 \
      --seq-len 5000

  uv run python src/analyze_revcomp_tokenization.py \
      --model dnagpt/human_gpt2-v1 \
      --suite gb \
      --max-samples 2000 \
      --seq-len 4096 \
      --output ./eval_results/revcomp_tokenization_gb.json

  uv run python src/analyze_revcomp_tokenization.py \
      --model dnagpt/human_gpt2-v1 \
      --suite nt \
      --max-samples 2000 \
      --seq-len 4096 \
      --output ./eval_results/revcomp_tokenization_nt.json
"""

import argparse
import csv
import gzip
import json
import os
import random
from collections import Counter
from statistics import mean, median

from transformers import AutoTokenizer


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BASES = "ACGT"

GB_TASKS = [
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "dummy_mouse_enhancers_ensembl",
]

NT_TASKS = [
    "H3K4me3",
    "H3K36me3",
    "H3K9ac",
    "splice_sites_all",
]

NT_HF_DATASET = "InstaDeepAI/nucleotide_transformer_downstream_tasks"


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def clean_dna(seq: str) -> str:
    seq = seq.strip().upper()
    return "".join(ch for ch in seq if ch in "ACGT")


def iter_fasta_chunks(path: str, seq_len: int, max_samples: int):
    opener = gzip.open if str(path).endswith(".gz") else open
    yielded = 0
    with opener(path, "rt") as fh:
        parts = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    seq = clean_dna("".join(parts))
                    for i in range(0, max(0, len(seq) - seq_len + 1), seq_len):
                        chunk = seq[i : i + seq_len]
                        if len(chunk) == seq_len:
                            yield chunk
                            yielded += 1
                            if yielded >= max_samples:
                                return
                    parts = []
                continue
            parts.append(line)

        if parts:
            seq = clean_dna("".join(parts))
            for i in range(0, max(0, len(seq) - seq_len + 1), seq_len):
                chunk = seq[i : i + seq_len]
                if len(chunk) == seq_len:
                    yield chunk
                    yielded += 1
                    if yielded >= max_samples:
                        return


def iter_gue_plus_epi(gue_plus_dir: str, seq_len: int, max_samples: int):
    epi_dir = os.path.join(gue_plus_dir, "EPI")
    subdirs = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
    yielded = 0

    for subdir in subdirs:
        path = os.path.join(epi_dir, subdir, "train.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "enhancer" in row and "promoter" in row:
                    seq = row["enhancer"] + row["promoter"]
                else:
                    seq = row.get("sequence") or row.get("seq") or ""
                seq = clean_dna(seq)
                if len(seq) >= seq_len:
                    yield seq[:seq_len]
                    yielded += 1
                    if yielded >= max_samples:
                        return


def iter_genomics_benchmarks(seq_len: int, max_samples: int, tasks: list[str] | None = None):
    from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset
    from genomic_benchmarks.loc2seq import download_dataset

    task_names = tasks or GB_TASKS
    per_task = max(1, (max_samples + len(task_names) - 1) // len(task_names))
    yielded = 0

    for task in task_names:
        download_dataset(task, version=0)
        emitted_for_task = 0
        for split in ("train", "test"):
            ds = GenomicClfDataset(task, split=split)
            for seq, _label in ds:
                seq = clean_dna(seq)
                if len(seq) >= seq_len:
                    yield seq[:seq_len]
                else:
                    yield seq
                yielded += 1
                emitted_for_task += 1
                if yielded >= max_samples or emitted_for_task >= per_task:
                    break
            if yielded >= max_samples or emitted_for_task >= per_task:
                break
        if yielded >= max_samples:
            return


def iter_nt_downstream(seq_len: int, max_samples: int, tasks: list[str] | None = None):
    from datasets import load_dataset

    task_names = set(tasks or NT_TASKS)
    per_task = max(1, (max_samples + len(task_names) - 1) // len(task_names))
    emitted = {task: 0 for task in task_names}
    yielded = 0

    ds = load_dataset(NT_HF_DATASET)
    for split in ("train", "test"):
        for row in ds[split]:
            task = row.get("task")
            if task not in task_names or emitted[task] >= per_task:
                continue
            seq = clean_dna(row["sequence"])
            if len(seq) >= seq_len:
                yield seq[:seq_len]
            else:
                yield seq
            emitted[task] += 1
            yielded += 1
            if yielded >= max_samples or all(n >= per_task for n in emitted.values()):
                return


def iter_random(seq_len: int, max_samples: int, seed: int):
    rng = random.Random(seed)
    for _ in range(max_samples):
        yield "".join(rng.choices(BASES, k=seq_len))


def token_boundaries(offsets, seq_len: int) -> set[int]:
    """Return internal BPE boundary positions from offset mappings."""
    boundaries = set()
    for start, end in offsets:
        if 0 < start < seq_len:
            boundaries.add(start)
        if 0 < end < seq_len:
            boundaries.add(end)
    return boundaries


def map_rc_boundaries_to_forward(offsets, seq_len: int) -> set[int]:
    """Map RC-token boundaries back into the original forward coordinates."""
    boundaries = set()
    for start, end in offsets:
        mapped_start = seq_len - end
        mapped_end = seq_len - start
        if 0 < mapped_start < seq_len:
            boundaries.add(mapped_start)
        if 0 < mapped_end < seq_len:
            boundaries.add(mapped_end)
    return boundaries


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def multiset_jaccard(a, b) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / max(1, union)


def encode_with_offsets(tokenizer, seq: str):
    enc = tokenizer(
        seq,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    return enc["input_ids"], enc["offset_mapping"]


def analyze_one(tokenizer, seq: str, idx: int):
    rc = reverse_complement(seq)
    ids, offsets = encode_with_offsets(tokenizer, seq)
    rc_ids, rc_offsets = encode_with_offsets(tokenizer, rc)

    boundaries = token_boundaries(offsets, len(seq))
    rc_boundaries = map_rc_boundaries_to_forward(rc_offsets, len(seq))

    id_set_j = jaccard(set(ids), set(rc_ids))
    id_multi_j = multiset_jaccard(ids, rc_ids)
    boundary_j = jaccard(boundaries, rc_boundaries)

    return {
        "idx": idx,
        "seq_len": len(seq),
        "n_tokens": len(ids),
        "n_tokens_rc": len(rc_ids),
        "token_count_delta": len(rc_ids) - len(ids),
        "token_count_ratio": len(rc_ids) / max(1, len(ids)),
        "boundary_jaccard": boundary_j,
        "token_id_set_jaccard": id_set_j,
        "token_id_multiset_jaccard": id_multi_j,
        "seq_prefix": seq[:120],
        "rc_prefix": rc[:120],
        "tokens_prefix": tokenizer.convert_ids_to_tokens(ids[:30]),
        "rc_tokens_prefix": tokenizer.convert_ids_to_tokens(rc_ids[:30]),
    }


def summarize(values):
    values = list(values)
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Analyze BPE tokenization under reverse complement.")
    p.add_argument("--model", default="dnagpt/human_gpt2-v1",
                   help="Tokenizer/model path or HF id.")

    source = p.add_mutually_exclusive_group()
    source.add_argument("--fasta", default=None, help="FASTA file to sample from.")
    source.add_argument("--gue-plus-dir", default=None,
                        help="GUE+ root containing EPI/ subdirectory.")
    source.add_argument("--suite", choices=("gb", "nt"), default=None,
                        help="Sample from built-in benchmark suites: gb or nt.")
    source.add_argument("--random", action="store_true",
                        help="Use random A/C/G/T sequences.")

    p.add_argument("--tasks", nargs="+", default=None,
                   help="Optional task names for --suite gb/nt.")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show-examples", type=int, default=5,
                   help="Number of worst-aligned examples to include.")
    p.add_argument("--output", default=None,
                   help="Optional JSON path for full summary and examples.")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.fasta:
        seq_iter = iter_fasta_chunks(args.fasta, args.seq_len, args.max_samples)
        source_name = f"fasta:{args.fasta}"
    elif args.gue_plus_dir:
        seq_iter = iter_gue_plus_epi(args.gue_plus_dir, args.seq_len, args.max_samples)
        source_name = f"gue_plus_epi:{args.gue_plus_dir}"
    elif args.suite == "gb":
        seq_iter = iter_genomics_benchmarks(args.seq_len, args.max_samples, args.tasks)
        source_name = "genomics_benchmarks:" + ",".join(args.tasks or GB_TASKS)
    elif args.suite == "nt":
        seq_iter = iter_nt_downstream(args.seq_len, args.max_samples, args.tasks)
        source_name = "nt_downstream:" + ",".join(args.tasks or NT_TASKS)
    else:
        seq_iter = iter_random(args.seq_len, args.max_samples, args.seed)
        source_name = "random"

    records = []
    for idx, seq in enumerate(seq_iter):
        records.append(analyze_one(tokenizer, seq, idx))

    if not records:
        raise RuntimeError("No sequences were loaded. Check input path and seq_len.")

    summary = {
        "model": args.model,
        "source": source_name,
        "n_samples": len(records),
        "seq_len": args.seq_len,
        "n_tokens": summarize(r["n_tokens"] for r in records),
        "n_tokens_rc": summarize(r["n_tokens_rc"] for r in records),
        "token_count_delta": summarize(r["token_count_delta"] for r in records),
        "token_count_ratio": summarize(r["token_count_ratio"] for r in records),
        "boundary_jaccard": summarize(r["boundary_jaccard"] for r in records),
        "token_id_set_jaccard": summarize(r["token_id_set_jaccard"] for r in records),
        "token_id_multiset_jaccard": summarize(r["token_id_multiset_jaccard"] for r in records),
    }

    worst = sorted(records, key=lambda r: r["boundary_jaccard"])[: args.show_examples]
    result = {"summary": summary, "worst_boundary_examples": worst}

    print("=" * 72)
    print("Reverse-complement tokenization alignment")
    print("=" * 72)
    print(f"Tokenizer : {args.model}")
    print(f"Source    : {source_name}")
    print(f"Samples   : {len(records):,}")
    print(f"Seq len   : {args.seq_len:,} bp")
    print()
    for key in [
        "n_tokens",
        "n_tokens_rc",
        "token_count_delta",
        "token_count_ratio",
        "boundary_jaccard",
        "token_id_set_jaccard",
        "token_id_multiset_jaccard",
    ]:
        s = summary[key]
        print(
            f"{key:24s}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
            f"min={s['min']:.4f}  max={s['max']:.4f}"
        )

    print()
    print("Interpretation hints:")
    print("  boundary_jaccard close to 1.0 => RC preserves BPE boundaries.")
    print("  boundary_jaccard close to 0.0 => RC strongly changes BPE boundaries.")
    print("  token_id_*_jaccard measures token identity overlap, not order.")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nSaved JSON: {args.output}")


if __name__ == "__main__":
    main()
