#!/usr/bin/env python3
"""
Aggregate DNAGPT RC-consistency results across M0-M6.

Expected usage on the remote machine:

    python aggregate_rc_consistent.py \
      --input-dir /gpfs/scratch/bty252/dna_foundation_model/figures/rc_consistency \
      --out-prefix /gpfs/scratch/bty252/dna_foundation_model/figures/rc_consistency/rc_consistent_summary

The script recursively scans JSON/CSV files under --input-dir. It supports
layouts like:

    rc_consistency/m0/*.json
    rc_consistency/m1/*.json
    ...
    rc_consistency/m6/*.json

It writes:

    <out-prefix>.csv      summary table
    <out-prefix>.md       Markdown summary table
    <out-prefix>.raw.csv  per-file/per-run extracted values
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any


MODEL_RE = re.compile(r"(?<![A-Za-z0-9])(M[0-6])(?![A-Za-z0-9])", re.IGNORECASE)
SEED_RE = re.compile(r"(?:seed|s)(\d+)", re.IGNORECASE)
REP_RE = re.compile(r"(?:repeat|rep|r)(\d+)", re.IGNORECASE)

DEFAULT_METRIC_CANDIDATES = [
    "rc_consistency",
    "rc_consistent",
    "rc_consistency_score",
    "rc_score",
    "consistency",
    "agreement",
    "same_pred_rate",
    "same_prediction_rate",
    "rc_agreement",
    "cosine",
    "mean_cosine",
    "avg_cosine",
    "pearson",
    "spearman",
]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, next_key))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_key = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten(value, next_key))
    else:
        out[prefix] = obj
    return out


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def infer_model(path: Path, row: dict[str, Any]) -> str:
    for key in ("model", "model_name", "name"):
        value = str(row.get(key, "")).strip()
        if value:
            match = MODEL_RE.search(value)
            return match.group(1).upper() if match else value

    for part in path.parts:
        match = MODEL_RE.search(part)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"


def infer_seed(path: Path, row: dict[str, Any]) -> str:
    for key in ("seed", "random_seed"):
        value = str(row.get(key, "")).strip()
        if value:
            return value

    match = SEED_RE.search(path.name)
    return match.group(1) if match else ""


def infer_repeat(path: Path, row: dict[str, Any]) -> str:
    for key in ("repeat", "repeat_index", "rep", "run"):
        value = str(row.get(key, "")).strip()
        if value:
            return value

    match = REP_RE.search(path.name)
    return match.group(1) if match else ""


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rows.append(flatten(item))
        return rows

    if isinstance(data, dict):
        rows.append(flatten(data))

        # Support common nested result layouts, e.g.
        # {"results": {"M0": {"rc_consistency": 0.9}, ...}}
        for container_key in ("results", "models", "overall", "summary"):
            container = data.get(container_key)
            if isinstance(container, dict):
                for model_name, metrics in container.items():
                    if isinstance(metrics, dict):
                        row = flatten(metrics)
                        row["model"] = model_name
                        rows.append(row)

    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_metric(rows: list[tuple[Path, dict[str, Any]]]) -> str | None:
    numeric_keys: dict[str, int] = {}
    for _path, row in rows:
        for key, value in row.items():
            if as_float(value) is not None:
                numeric_keys[key] = numeric_keys.get(key, 0) + 1

    for candidate in DEFAULT_METRIC_CANDIDATES:
        matches = [
            key
            for key in numeric_keys
            if key.lower() == candidate.lower() or key.lower().endswith(f".{candidate.lower()}")
        ]
        if matches:
            return sorted(matches, key=lambda key: numeric_keys[key], reverse=True)[0]

    if not numeric_keys:
        return None

    return sorted(numeric_keys, key=lambda key: numeric_keys[key], reverse=True)[0]


def fmt(value: float) -> str:
    return f"{value:.4f}"


def model_sort_key(model: str) -> tuple[int, int | str]:
    match = MODEL_RE.fullmatch(model)
    if match:
        return (0, int(match.group(1)[1:]))
    if model == "UNKNOWN":
        return (2, model)
    return (1, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RC-consistency results across M0-M6.")
    parser.add_argument("--input-dir", required=True, help="Parent directory containing m0/... m6/... results.")
    parser.add_argument("--metric", default=None, help="Metric key to aggregate. If omitted, inferred automatically.")
    parser.add_argument("--out-prefix", default="rc_consistent_summary", help="Output path prefix.")
    parser.add_argument("--list-metrics", action="store_true", help="Print numeric metric candidates and exit.")
    parser.add_argument("--sample-std", action="store_true", help="Use sample std instead of population std.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    files = sorted(list(root.rglob("*.json")) + list(root.rglob("*.csv")))

    all_rows: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        try:
            rows = load_json_rows(path) if path.suffix.lower() == ".json" else load_csv_rows(path)
            for row in rows:
                all_rows.append((path, row))
        except Exception as exc:
            print(f"[warn] skipping {path}: {exc}")

    if not all_rows:
        raise SystemExit(f"No JSON/CSV rows found under {root}")

    numeric_keys = sorted(
        {
            key
            for _path, row in all_rows
            for key, value in row.items()
            if as_float(value) is not None
        }
    )

    if args.list_metrics:
        print("Numeric metric candidates:")
        for key in numeric_keys:
            print(f"  {key}")
        return

    metric = args.metric or discover_metric(all_rows)
    if not metric:
        raise SystemExit("Could not infer metric. Re-run with --list-metrics, then pass --metric KEY.")

    records: list[dict[str, Any]] = []
    for path, row in all_rows:
        value = as_float(row.get(metric))
        if value is None:
            continue

        records.append(
            {
                "model": infer_model(path, row),
                "seed": infer_seed(path, row),
                "repeat": infer_repeat(path, row),
                "metric": metric,
                "value": value,
                "file": str(path),
            }
        )

    if not records:
        raise SystemExit(f"No values found for metric: {metric}")

    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)

    baseline_mean = None
    if "M0" in by_model:
        baseline_mean = mean(record["value"] for record in by_model["M0"])

    summary: list[dict[str, Any]] = []
    for model in sorted(by_model, key=model_sort_key):
        values = [record["value"] for record in by_model[model]]
        std = stdev(values) if args.sample_std and len(values) > 1 else pstdev(values)
        avg = mean(values)
        summary.append(
            {
                "model": model,
                "n": len(values),
                "mean": avg,
                "std": std,
                "delta_vs_M0": "" if baseline_mean is None else avg - baseline_mean,
            }
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")
    out_raw = out_prefix.with_suffix(".raw.csv")

    with out_raw.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed", "repeat", "metric", "value", "file"])
        writer.writeheader()
        writer.writerows(records)

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "n", "mean", "std", "delta_vs_M0"])
        writer.writeheader()
        writer.writerows(summary)

    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("# RC-Consistency Summary\n\n")
        handle.write(f"Metric: `{metric}`\n\n")
        handle.write("| Model | n | mean +/- std | Delta vs M0 |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in summary:
            delta = "" if row["delta_vs_M0"] == "" else fmt(float(row["delta_vs_M0"]))
            handle.write(
                f"| {row['model']} | {row['n']} | "
                f"{fmt(float(row['mean']))} +/- {fmt(float(row['std']))} | {delta} |\n"
            )

    print(f"Metric: {metric}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_raw}")


if __name__ == "__main__":
    main()
