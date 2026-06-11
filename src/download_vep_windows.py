#!/usr/bin/env python
"""
Download Genomics Long-Range Benchmark VEP windows as a flat CSV.

The Caduceus VEP experiment uses the Genomics LRB eQTL SNP task. The upstream
dataset is implemented as a Hugging Face dataset script, so it requires
`datasets<4` plus `trust_remote_code=True`.

Example:
  uv run python src/download_vep_windows.py \
    --sequence-length 4096 \
    --output ./data/vep_windows.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections.abc import Mapping


DATASET_NAME = "InstaDeepAI/genomics-long-range-benchmark"
TASK_FALLBACKS = ("variant_effect_causal_eqtl", "variant_effect_gene_expression")


def pick(row: Mapping, names: tuple[str, ...]):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of {names} found. Available columns: {list(row.keys())}")


def prepare_reference_genome(reference_genome: str | None) -> None:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    datasets_cache = os.environ.get("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    downloads_dir = os.path.join(datasets_cache, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    if not reference_genome:
        return

    source = os.path.abspath(reference_genome)
    if not os.path.isfile(source):
        raise SystemExit(f"--reference-genome does not exist: {source}")

    target = os.path.join(downloads_dir, "hg38.fa")
    if os.path.exists(target):
        return

    try:
        os.symlink(source, target)
    except OSError:
        shutil.copyfile(source, target)


def load_lrb_dataset(task_name: str | None, sequence_length: int, reference_genome: str | None):
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `datasets>=3.5,<4`.") from exc

    version = tuple(int(part) for part in datasets.__version__.split(".")[:2])
    if version >= (4, 0):
        raise SystemExit(
            f"datasets=={datasets.__version__} cannot load Hugging Face dataset scripts. "
            "Install a 3.x release, e.g. `uv pip install 'datasets>=3.5,<4'`, "
            "then run this command again."
        )

    prepare_reference_genome(reference_genome)

    task_names = (task_name,) if task_name else TASK_FALLBACKS
    errors = []
    for candidate in task_names:
        try:
            return load_dataset(
                DATASET_NAME,
                task_name=candidate,
                sequence_length=sequence_length,
                trust_remote_code=True,
            ), candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Could not load any VEP task:\n" + "\n".join(errors))


def parse_args():
    parser = argparse.ArgumentParser(description="Export Genomics LRB VEP windows to CSV.")
    parser.add_argument("--output", default="./data/vep_windows.csv")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument(
        "--reference-genome",
        default=None,
        help="Optional local hg38 FASTA path, e.g. ./data/hg38.fa, to avoid re-downloading hg38.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help=(
            "Optional LRB task name. Defaults to trying variant_effect_causal_eqtl "
            "then variant_effect_gene_expression."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, task_name = load_lrb_dataset(args.task_name, args.sequence_length, args.reference_genome)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_rows = 0
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ref_sequence",
                "alt_sequence",
                "label",
                "distance_to_tss",
                "split",
                "tissue",
                "chromosome",
                "position",
            ],
        )
        writer.writeheader()

        for split_name, split in dataset.items():
            if len(split) == 0:
                continue
            for row in split:
                writer.writerow(
                    {
                        "ref_sequence": pick(
                            row,
                            ("ref_forward_sequence", "ref_sequence", "ref", "reference_sequence"),
                        ),
                        "alt_sequence": pick(
                            row,
                            ("alt_forward_sequence", "alt_sequence", "alt", "alternate_sequence"),
                        ),
                        "label": pick(row, ("labels", "label")),
                        "distance_to_tss": pick(
                            row,
                            (
                                "distance_to_nearest_tss",
                                "distance_to_nearest_TSS",
                                "distance_to_tss",
                                "distance",
                            ),
                        ),
                        "split": split_name,
                        "tissue": row.get("tissue", ""),
                        "chromosome": row.get("chromosome", ""),
                        "position": row.get("position", ""),
                    }
                )
                n_rows += 1

    print(f"Loaded task: {task_name}")
    print(dataset)
    print(f"Wrote {n_rows:,} rows to {args.output}")


if __name__ == "__main__":
    main()
