"""
Prepare a coordinate manifest for long-context HyenaDNA contrastive training.

The manifest stores stable window IDs and 0-based half-open coordinates. It does
not store sequences, so later stages can sample positives/negatives by ID and
fetch sequence from the reference FASTA on demand.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


DEFAULT_CHROMS = [f"chr{i}" for i in range(1, 23)]
DEFAULT_VALID_CHROMS = {"chr8", "chr9"}


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def fasta_records(path: str):
    name = None
    chunks = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks).upper()


def count_non_acgt(seq: str) -> int:
    return sum(base not in "ACGT" for base in seq)


def split_for_chrom(chrom: str, valid_chroms: set[str], test_chroms: set[str]) -> str:
    if chrom in test_chroms:
        return "test"
    if chrom in valid_chroms:
        return "valid"
    return "train"


def make_window_id(prefix: str, chrom: str, start: int, end: int) -> str:
    return f"{prefix}_{chrom}_{start:09d}_{end:09d}"


def iter_windows(seq_len: int, window_bp: int, stride_bp: int, include_tail: bool):
    if seq_len < window_bp:
        return
    start = 0
    last_start = seq_len - window_bp
    while start <= last_start:
        yield start, start + window_bp
        start += stride_bp
    if include_tail and start - stride_bp < last_start:
        yield last_start, seq_len


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-fasta", required=True, help="Reference FASTA, e.g. hg38.fa or hg38.fa.gz.")
    parser.add_argument("--output", required=True, help="Output TSV manifest.")
    parser.add_argument("--window-bp", type=int, default=450000)
    parser.add_argument("--stride-bp", type=int, default=0, help="0 means non-overlapping windows.")
    parser.add_argument("--include-tail", action="store_true", help="Include one final tail-aligned window per chrom.")
    parser.add_argument("--chroms", nargs="+", default=DEFAULT_CHROMS)
    parser.add_argument("--valid-chroms", nargs="+", default=sorted(DEFAULT_VALID_CHROMS))
    parser.add_argument("--test-chroms", nargs="+", default=[])
    parser.add_argument(
        "--no-valid-split",
        action="store_true",
        help="Assign every kept window to train, ignoring --valid-chroms and --test-chroms.",
    )
    parser.add_argument("--max-n-fraction", type=float, default=0.0)
    parser.add_argument("--max-windows-per-chrom", type=int, default=0, help="0 means no per-chrom limit.")
    parser.add_argument("--id-prefix", default="hg38_450k")
    args = parser.parse_args()

    if args.window_bp <= 0:
        raise ValueError("--window-bp must be positive.")
    stride_bp = args.window_bp if args.stride_bp == 0 else args.stride_bp
    if stride_bp <= 0:
        raise ValueError("--stride-bp must be positive or 0.")

    chroms = set(args.chroms)
    valid_chroms = set(args.valid_chroms)
    test_chroms = set(args.test_chroms)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "chrom",
        "start",
        "end",
        "length",
        "split",
        "n_count",
        "n_frac",
    ]
    total = 0
    kept = 0
    skipped_n = 0
    skipped_chrom = 0
    split_counts: dict[str, int] = {}
    chrom_counts: dict[str, int] = {}

    with open(output, "w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for chrom, seq in fasta_records(args.genome_fasta):
            if chrom not in chroms:
                skipped_chrom += 1
                continue
            per_chrom = 0
            for start, end in iter_windows(len(seq), args.window_bp, stride_bp, args.include_tail):
                total += 1
                if args.max_windows_per_chrom and per_chrom >= args.max_windows_per_chrom:
                    continue
                window = seq[start:end]
                n_count = count_non_acgt(window)
                n_frac = n_count / max(len(window), 1)
                if n_frac > args.max_n_fraction:
                    skipped_n += 1
                    continue
                split = "train" if args.no_valid_split else split_for_chrom(chrom, valid_chroms, test_chroms)
                row = {
                    "id": make_window_id(args.id_prefix, chrom, start, end),
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "split": split,
                    "n_count": n_count,
                    "n_frac": f"{n_frac:.6g}",
                }
                writer.writerow(row)
                kept += 1
                per_chrom += 1
                split_counts[split] = split_counts.get(split, 0) + 1
                chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1

    print(f"Output                  : {output}")
    print(f"Window bp / stride bp   : {args.window_bp:,} / {stride_bp:,}")
    print(f"No valid/test split      : {args.no_valid_split}")
    print(f"Candidate windows       : {total:,}")
    print(f"Kept windows            : {kept:,}")
    print(f"Skipped for N fraction  : {skipped_n:,}")
    print(f"Skipped FASTA records   : {skipped_chrom:,}")
    print(f"Split counts            : {split_counts}")
    print(f"Chrom counts            : {chrom_counts}")


if __name__ == "__main__":
    main()
