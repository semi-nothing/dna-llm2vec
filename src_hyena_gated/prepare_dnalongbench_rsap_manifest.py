"""
Prepare a DNALONGBENCH RSAP coordinate manifest.

RSAP is regulatory sequence activity prediction from long genomic windows. The
script is schema-tolerant because released activity tables may store one target
column or many assay/tissue/cell-type activity columns.

Output columns consumed by train_dnalongbench_rsap.py:
  chrom, start, end, split, strand, region_id, reference_point, <label columns...>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
from typing import Iterable


DEFAULT_EXCLUDE_COLUMNS = {
    "chrom",
    "chr",
    "chromosome",
    "gene_chrom",
    "gene_start",
    "gene_end",
    "start",
    "end",
    "strand",
    "gene_strand",
    "gene_id",
    "region_id",
    "gene_name",
    "gene_type",
    "transcript_id",
    "subset",
    "split",
    "partition",
    "set",
    "tss",
    "gene_tss",
}


def open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def sniff_dialect(path: str) -> csv.Dialect:
    with open_text(path, "r") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        return csv.excel_tab if path.endswith((".tsv", ".tsv.gz")) else csv.excel


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"valid", "val", "dev", "validation"}:
        return "valid"
    if value in {"test", "testing"}:
        return "test"
    raise ValueError(f"Unrecognized split: {value!r}")


def _get_first(row: dict[str, str], names: Iterable[str]) -> str | None:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and value != "":
            return value
    return None


def parse_float(value: str) -> float | None:
    value = str(value).strip()
    if value == "" or value.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def infer_label_columns(fieldnames: list[str]) -> list[str]:
    labels = []
    for name in fieldnames:
        if name.lower() in DEFAULT_EXCLUDE_COLUMNS:
            continue
        labels.append(name)
    if not labels:
        raise ValueError(
            "Could not infer expression label columns. Pass --label-columns explicitly."
        )
    return labels


def build_window(
    start: int,
    end: int,
    strand: str,
    length: int,
    mode: str,
) -> tuple[int, int, int]:
    if mode == "preserve":
        return start, end, (start + end) // 2
    if mode == "center_resize":
        center = (start + end) // 2
    elif mode == "tss":
        center = end if strand == "-" else start
    elif mode == "gene_midpoint":
        center = (start + end) // 2
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    window_start = center - length // 2
    window_end = window_start + length
    if window_start < 0:
        window_end -= window_start
        window_start = 0
    return window_start, window_end, center


def main():
    parser = argparse.ArgumentParser(description="Prepare DNALONGBENCH RSAP coordinate manifest")
    parser.add_argument("--targets", required=True, help="DNALONGBENCH RSAP activity table.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=450000)
    parser.add_argument("--label-columns", nargs="+", default=None)
    parser.add_argument("--chrom-column", default="chrom")
    parser.add_argument("--start-column", default="start")
    parser.add_argument("--end-column", default="end")
    parser.add_argument("--strand-column", default="gene_strand")
    parser.add_argument("--region-id-column", default="region_id")
    parser.add_argument("--split-column", default="subset")
    parser.add_argument(
        "--window-mode",
        choices=("preserve", "center_resize", "tss", "gene_midpoint"),
        default="preserve",
        help=(
            "How to derive the output window. 'preserve' keeps input coordinates; "
            "'center_resize' resizes around the input interval midpoint."
        ),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dialect = sniff_dialect(args.targets)
    total = 0
    kept = 0
    skipped_missing_label = 0
    splits: dict[str, int] = {}

    with open_text(args.targets, "r") as src:
        reader = csv.DictReader(src, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Input table has no header: {args.targets}")
        label_columns = args.label_columns or infer_label_columns(reader.fieldnames)
        fieldnames = [
            "chrom",
            "start",
            "end",
            "split",
            "strand",
            "region_id",
            "reference_point",
            *label_columns,
        ]

        with open_text(args.output, "w") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for row in reader:
                total += 1
                chrom = _get_first(row, [args.chrom_column, "chrom", "chr", "chromosome"])
                start_raw = _get_first(row, [args.start_column, "gene_start", "begin"])
                end_raw = _get_first(row, [args.end_column, "gene_end", "stop"])
                split_raw = _get_first(row, [args.split_column, "split", "partition", "set"])
                if chrom is None or start_raw is None or end_raw is None or split_raw is None:
                    raise ValueError(
                        "RSAP table must contain chromosome, start/end, and split columns. "
                        f"Available columns: {reader.fieldnames}"
                    )

                labels = {name: parse_float(row.get(name, "")) for name in label_columns}
                if any(value is None for value in labels.values()):
                    skipped_missing_label += 1
                    continue

                strand = _get_first(row, [args.strand_column, "strand"]) or "+"
                start, end, reference_point = build_window(
                    start=int(float(start_raw)),
                    end=int(float(end_raw)),
                    strand=strand,
                    length=args.length,
                    mode=args.window_mode,
                )
                split = normalize_split(split_raw)
                splits[split] = splits.get(split, 0) + 1
                kept += 1

                writer.writerow(
                    {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "split": split,
                        "strand": "-" if strand == "-" else "+",
                        "region_id": _get_first(row, [args.region_id_column, "gene_id", "gene_name"]) or "",
                        "reference_point": reference_point,
                        **{name: labels[name] for name in label_columns},
                    }
                )

    print(f"Input rows                : {total:,}")
    print(f"Kept rows                 : {kept:,}")
    print(f"Skipped missing labels    : {skipped_missing_label:,}")
    print(f"Activity targets          : {label_columns}")
    print(f"Splits                    : {splits}")
    print(f"Output                    : {args.output}")


if __name__ == "__main__":
    main()
