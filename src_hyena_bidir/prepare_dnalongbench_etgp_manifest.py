"""
Prepare a DNALONGBENCH ETGP coordinate manifest.

Input target TSV columns observed in the DNALONGBENCH ETGP data:
  gene_chrom, gene_start, gene_end, region_chrom, region_start, region_end,
  gene_id, gene_strand, region_id, target, subset, ...

Output columns consumed by train_dnalongbench_etgp.py:
  chrom, start, end, label, split, strand, gene_id, region_id, distance_to_tss
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os


def open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def parse_label(value: str) -> int:
    value = value.strip().lower()
    if value in {"positive", "pos", "1", "true", "yes"}:
        return 1
    if value in {"negative", "neg", "0", "false", "no"}:
        return 0
    raise ValueError(f"Unrecognized ETGP target label: {value!r}")


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"valid", "val", "dev", "validation"}:
        return "valid"
    if value in {"test", "testing"}:
        return "test"
    raise ValueError(f"Unrecognized split: {value!r}")


def tss_from_gene(gene_start: int, gene_end: int, strand: str) -> int:
    if strand == "-":
        return gene_end
    return gene_start


def build_window(
    gene_start: int,
    gene_end: int,
    region_start: int,
    region_end: int,
    strand: str,
    length: int,
    mode: str,
) -> tuple[int, int, int]:
    tss = tss_from_gene(gene_start, gene_end, strand)
    region_center = (region_start + region_end) // 2

    if mode == "pair_midpoint":
        center = (tss + region_center) // 2
    elif mode == "tss":
        center = tss
    elif mode == "span_then_center":
        left = min(tss, region_start)
        right = max(tss, region_end)
        center = (left + right) // 2
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    start = center - length // 2
    end = start + length
    if start < 0:
        end -= start
        start = 0
    return start, end, tss


def main():
    parser = argparse.ArgumentParser(description="Prepare DNALONGBENCH ETGP coordinate manifest")
    parser.add_argument("--targets", required=True, help="DNALONGBENCH ETGP *.data.tsv file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=450000)
    parser.add_argument(
        "--window-mode",
        choices=("pair_midpoint", "tss", "span_then_center"),
        default="pair_midpoint",
        help="How to place the fixed-length genomic window around the enhancer/TSS pair.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total = 0
    kept = 0
    skipped_cross_chrom = 0
    labels = {0: 0, 1: 0}
    splits: dict[str, int] = {}

    with open_text(args.targets, "r") as src, open_text(args.output, "w") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        fieldnames = [
            "chrom",
            "start",
            "end",
            "label",
            "split",
            "strand",
            "gene_id",
            "region_id",
            "gene_tss",
            "region_start",
            "region_end",
            "distance_to_tss",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            total += 1
            gene_chrom = row["gene_chrom"]
            region_chrom = row["region_chrom"]
            if gene_chrom != region_chrom:
                skipped_cross_chrom += 1
                continue

            gene_start = int(row["gene_start"])
            gene_end = int(row["gene_end"])
            region_start = int(row["region_start"])
            region_end = int(row["region_end"])
            strand = row.get("gene_strand", "+").strip() or "+"
            start, end, tss = build_window(
                gene_start=gene_start,
                gene_end=gene_end,
                region_start=region_start,
                region_end=region_end,
                strand=strand,
                length=args.length,
                mode=args.window_mode,
            )
            label = parse_label(row["target"])
            split = normalize_split(row["subset"])
            labels[label] += 1
            splits[split] = splits.get(split, 0) + 1
            kept += 1

            writer.writerow(
                {
                    "chrom": gene_chrom,
                    "start": start,
                    "end": end,
                    "label": label,
                    "split": split,
                    "strand": "-" if strand == "-" else "+",
                    "gene_id": row.get("gene_id", ""),
                    "region_id": row.get("region_id", ""),
                    "gene_tss": tss,
                    "region_start": region_start,
                    "region_end": region_end,
                    "distance_to_tss": row.get("distance_to_tss", ""),
                }
            )

    print(f"Input rows                : {total:,}")
    print(f"Kept rows                 : {kept:,}")
    print(f"Skipped cross-chrom rows  : {skipped_cross_chrom:,}")
    print(f"Labels                    : negative={labels[0]:,}, positive={labels[1]:,}")
    print(f"Splits                    : {splits}")
    print(f"Output                    : {args.output}")


if __name__ == "__main__":
    main()
