#!/usr/bin/env python3
"""
Aggregate DNAGPT RC-consistency outputs written by bp_mutagenesis_jacobian_dna.py.

This script is for results like:

    /gpfs/scratch/bty252/dna_foundation_model/figures/rc_consistency/
      m0/**/*_rc_compare.npz
      m1/**/*_rc_compare.npz
      ...
      m6/**/*_rc_compare.npz

Each *_rc_compare.npz contains:

    corr                  scalar correlation between forward and RC-flipped contact maps
    contact_aligned       forward contact map after coordinate alignment
    rc_flipped_aligned    reverse-complement contact map after flipping/alignment
    forward_positions     aligned forward bp positions
    rc_positions          aligned RC bp positions

Example:

    python aggregate_rc_consistent.py \
      --input-dir /gpfs/scratch/bty252/dna_foundation_model/figures/rc_consistency \
      --out-prefix /gpfs/scratch/bty252/dna_foundation_model/figures/rc_consistency/rc_consistent_summary

Outputs:

    <out-prefix>.csv      per-model mean/std summary
    <out-prefix>.md       Markdown table
    <out-prefix>.raw.csv  one row per *_rc_compare.npz
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any

import numpy as np


MODEL_RE = re.compile(r"(?<![A-Za-z0-9])([MH][0-6])(?![A-Za-z0-9])", re.IGNORECASE)
SEED_RE = re.compile(r"(?<![A-Za-z0-9])(?:seed|s)(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
REP_RE = re.compile(r"(?<![A-Za-z0-9])(?:repeat|rep|r)(\d+)(?![A-Za-z0-9])", re.IGNORECASE)


def as_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        out = float(arr.reshape(-1)[0])
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def infer_model(path: Path) -> str:
    for part in path.parts:
        match = MODEL_RE.fullmatch(part)
        if match:
            return match.group(1).upper()

    for part in path.parts:
        match = MODEL_RE.search(part)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"


def infer_seed(path: Path) -> str:
    match = SEED_RE.search(path.name)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        match = SEED_RE.search(part)
        if match:
            return match.group(1)
    return ""


def infer_repeat(path: Path) -> str:
    match = REP_RE.search(path.name)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        match = REP_RE.search(part)
        if match:
            return match.group(1)
    return ""


def infer_fragment(path: Path) -> str:
    name = path.name
    if name.endswith("_rc_compare.npz"):
        return name[: -len("_rc_compare.npz")]
    return path.stem


def model_sort_key(model: str) -> tuple[int, int | str]:
    match = MODEL_RE.fullmatch(model)
    if match:
        prefix = 0 if match.group(1)[0].upper() == "M" else 1
        return (prefix, int(match.group(1)[1:]))
    if model == "UNKNOWN":
        return (3, model)
    return (2, model)


def collect_files(root: Path, model_family: str) -> list[Path]:
    if model_family == "dnagpt":
        allowed = re.compile(r"m[0-6]", re.IGNORECASE)
    elif model_family == "hyena":
        allowed = re.compile(r"h[0-6]", re.IGNORECASE)
    elif model_family == "all":
        return sorted(root.rglob("*_rc_compare.npz"))
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    files: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and allowed.fullmatch(child.name):
            files.extend(child.rglob("*_rc_compare.npz"))
    return sorted(files)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def list_metrics(files: list[Path]) -> None:
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}

    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                for key in data.files:
                    if as_float(data[key]) is not None:
                        counts[key] = counts.get(key, 0) + 1
                        examples.setdefault(key, str(np.asarray(data[key]).reshape(-1)[0]))
        except Exception as exc:
            print(f"[warn] skipping {path}: {exc}")

    print("Scalar numeric metric candidates in *_rc_compare.npz:")
    for key in sorted(counts):
        print(f"  {key}  ({counts[key]} files; example={examples[key]})")


def load_record(path: Path, metric: str) -> dict[str, Any] | None:
    with np.load(path, allow_pickle=True) as data:
        if metric not in data.files:
            return None

        value = as_float(data[metric])
        if value is None:
            return None

        n_aligned = ""
        if "forward_positions" in data.files:
            n_aligned = int(np.asarray(data["forward_positions"]).shape[0])
        elif "contact_aligned" in data.files:
            n_aligned = int(np.asarray(data["contact_aligned"]).shape[0])

        contact_shape = ""
        if "contact_aligned" in data.files:
            contact_shape = "x".join(str(x) for x in np.asarray(data["contact_aligned"]).shape)

    return {
        "model": infer_model(path),
        "seed": infer_seed(path),
        "repeat": infer_repeat(path),
        "fragment": infer_fragment(path),
        "metric": metric,
        "value": value,
        "n_aligned_positions": n_aligned,
        "contact_shape": contact_shape,
        "file": str(path),
    }


def write_outputs(records: list[dict[str, Any]], sample_std: bool, out_prefix: Path, metric: str) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)

    baseline_model = None
    for candidate in ("M0", "H0"):
        if candidate in by_model:
            baseline_model = candidate
            break
    baseline_mean = None if baseline_model is None else mean(record["value"] for record in by_model[baseline_model])

    summary: list[dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        values = [record["value"] for record in by_model[model]]
        avg = mean(values)
        std = stdev(values) if sample_std and len(values) > 1 else pstdev(values)
        n_positions = [
            record["n_aligned_positions"]
            for record in by_model[model]
            if isinstance(record["n_aligned_positions"], int)
        ]
        summary.append(
            {
                "model": model,
                "n_fragments": len(values),
                "mean": avg,
                "std": std,
                "delta_vs_baseline": "" if baseline_mean is None else avg - baseline_mean,
                "mean_aligned_positions": "" if not n_positions else mean(n_positions),
            }
        )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")
    out_raw = out_prefix.with_suffix(".raw.csv")

    with out_raw.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "seed",
                "repeat",
                "fragment",
                "metric",
                "value",
                "n_aligned_positions",
                "contact_shape",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "n_fragments", "mean", "std", "delta_vs_baseline", "mean_aligned_positions"],
        )
        writer.writeheader()
        writer.writerows(summary)

    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("# RC-Consistency Summary\n\n")
        handle.write(f"Metric: `{metric}` from `*_rc_compare.npz`\n\n")
        baseline_label = baseline_model or "baseline"
        handle.write(f"| Model | fragments | mean +/- std | Delta vs {baseline_label} | mean aligned positions |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in summary:
            delta = "" if row["delta_vs_baseline"] == "" else fmt(float(row["delta_vs_baseline"]))
            mean_pos = "" if row["mean_aligned_positions"] == "" else fmt(float(row["mean_aligned_positions"]))
            handle.write(
                f"| {row['model']} | {row['n_fragments']} | "
                f"{fmt(float(row['mean']))} +/- {fmt(float(row['std']))} | "
                f"{delta} | {mean_pos} |\n"
            )

    print(f"Metric: {metric}")
    print(f"Records: {len(records)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_raw}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate *_rc_compare.npz RC-consistency results.")
    parser.add_argument("--input-dir", required=True, help="Parent directory containing m0/... m6/... results.")
    parser.add_argument("--metric", default="corr", help="Scalar key inside *_rc_compare.npz (default: corr).")
    parser.add_argument("--out-prefix", default="rc_consistent_summary", help="Output path prefix.")
    parser.add_argument(
        "--model-family",
        choices=("dnagpt", "hyena", "all"),
        default="dnagpt",
        help="Which top-level model folders to scan: dnagpt=m0-m6, hyena=h0-h6, all=everything.",
    )
    parser.add_argument("--list-metrics", action="store_true", help="Print scalar numeric npz keys and exit.")
    parser.add_argument("--sample-std", action="store_true", help="Use sample std instead of population std.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    files = collect_files(root, args.model_family)

    if not files:
        raise SystemExit(f"No *_rc_compare.npz files found under {root} for --model-family {args.model_family}")

    if args.list_metrics:
        list_metrics(files)
        return

    records: list[dict[str, Any]] = []
    for path in files:
        try:
            record = load_record(path, args.metric)
            if record is not None:
                records.append(record)
        except Exception as exc:
            print(f"[warn] skipping {path}: {exc}")

    if not records:
        raise SystemExit(f"No scalar values found for metric {args.metric!r} under {root}")

    write_outputs(records, args.sample_std, Path(args.out_prefix), args.metric)


if __name__ == "__main__":
    main()
