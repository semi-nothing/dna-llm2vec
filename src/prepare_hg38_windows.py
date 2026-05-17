"""
Prepare fixed-length genomic windows from a multi-FASTA genome file.

This creates one small FASTA per sampled window plus a manifest CSV. The output
FASTA files can be passed directly to bp_mutagenesis_jacobian_dna.py via --fasta.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def fasta_records(path: str):
    name = None
    chunks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
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


def valid_window(seq: str, start: int, length: int, max_n_fraction: float):
    window = seq[start : start + length]
    if len(window) != length:
        return False
    n_frac = sum(base not in "ACGT" for base in window) / max(len(window), 1)
    return n_frac <= max_n_fraction


def sample_starts(seq: str, length: int, n: int, rng: np.random.Generator, max_n_fraction: float, max_tries: int):
    if len(seq) < length:
        return []
    starts = []
    seen = set()
    max_start = len(seq) - length
    tries = 0
    while len(starts) < n and tries < max_tries:
        tries += 1
        start = int(rng.integers(0, max_start + 1))
        if start in seen:
            continue
        seen.add(start)
        if valid_window(seq, start, length, max_n_fraction):
            starts.append(start)
    return sorted(starts)


def write_fasta(path: str, name: str, sequence: str, width: int = 80):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f">{name}\n")
        for i in range(0, len(sequence), width):
            fh.write(sequence[i : i + width] + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genome-fasta", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window-bp", type=int, default=4096)
    p.add_argument("--windows-per-chrom", type=int, default=5)
    p.add_argument("--chroms", nargs="+", default=None, help="Optional chromosome names to include, e.g. chr1 chr2")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-n-fraction", type=float, default=0.0)
    p.add_argument("--max-tries-per-chrom", type=int, default=10000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fasta_dir = out_dir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    chrom_filter = set(args.chroms) if args.chroms else None
    rng = np.random.default_rng(args.seed)
    rows = []

    for chrom, seq in fasta_records(args.genome_fasta):
        if chrom_filter is not None and chrom not in chrom_filter:
            continue
        starts = sample_starts(
            seq,
            args.window_bp,
            args.windows_per_chrom,
            rng,
            args.max_n_fraction,
            args.max_tries_per_chrom,
        )
        for idx, start in enumerate(starts):
            end = start + args.window_bp
            window_id = f"{chrom}_{start + 1}_{end}"
            fasta_path = fasta_dir / f"{window_id}.fa"
            write_fasta(str(fasta_path), window_id, seq[start:end])
            rows.append({
                "id": window_id,
                "chrom": chrom,
                "start_1based": start + 1,
                "end_1based": end,
                "length": args.window_bp,
                "fasta": str(fasta_path),
            })

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "chrom", "start_1based", "end_1based", "length", "fasta"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest}")
    print(f"Wrote {len(rows)} FASTA windows under {fasta_dir}")


if __name__ == "__main__":
    main()
