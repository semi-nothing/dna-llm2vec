"""
Prepare explicit negative samples for long-window contrastive training.

Input is the coordinate manifest produced by prepare_hg38_long_windows_manifest.py.
The output stores stable anchor-to-negative IDs, so a small physical batch can
still use a fixed pool of negatives during 1M HyenaDNA Step 3 training.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_manifest(path: str, split: str) -> list[dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "chrom", "start", "end", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    return rows


def interval_gap(a: dict[str, str], b: dict[str, str]) -> int:
    a_start, a_end = int(a["start"]), int(a["end"])
    b_start, b_end = int(b["start"]), int(b["end"])
    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0


def sample_without_replacement(
    rng: random.Random,
    pool: list[dict[str, str]],
    n: int,
) -> list[dict[str, str]]:
    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def choose_negatives(
    anchor: dict[str, str],
    rows: list[dict[str, str]],
    by_chrom: dict[str, list[dict[str, str]]],
    rng: random.Random,
    n_negatives: int,
    min_same_chrom_distance: int,
) -> tuple[list[dict[str, str]], str]:
    anchor_id = anchor["id"]
    anchor_chrom = anchor["chrom"]
    chosen: list[dict[str, str]] = []
    chosen_ids: set[str] = set()
    sources = []

    def add_from(pool: list[dict[str, str]], source: str) -> None:
        nonlocal chosen
        remaining = n_negatives - len(chosen)
        if remaining <= 0:
            return
        pool = [row for row in pool if row["id"] != anchor_id and row["id"] not in chosen_ids]
        picked = sample_without_replacement(rng, pool, remaining)
        for row in picked:
            chosen.append(row)
            chosen_ids.add(row["id"])
        if picked:
            sources.append(source)

    different_chrom = [row for row in rows if row["chrom"] != anchor_chrom]
    add_from(different_chrom, "different_chrom")

    same_chrom_far = [
        row
        for row in by_chrom[anchor_chrom]
        if row["id"] != anchor_id and interval_gap(anchor, row) >= min_same_chrom_distance
    ]
    add_from(same_chrom_far, "same_chrom_far")

    fallback = [row for row in rows if row["id"] != anchor_id]
    add_from(fallback, "same_split_fallback")

    return chosen, "+".join(sources) if sources else "none"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-manifest", required=True, help="Input long-window TSV manifest.")
    parser.add_argument("--output", required=True, help="Output negative TSV manifest.")
    parser.add_argument("--split", default="train", help="Split to sample within.")
    parser.add_argument("--n-negatives", type=int, default=10, help="Negatives to store per anchor.")
    parser.add_argument(
        "--min-same-chrom-distance",
        type=int,
        default=1_000_000,
        help="Minimum interval gap in bp for same-chromosome fallback negatives.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Write rows even when fewer than --n-negatives unique candidates are available.",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="For smoke tests only. 0 means all anchors in the selected split.",
    )
    args = parser.parse_args()

    if args.n_negatives <= 0:
        raise ValueError("--n-negatives must be positive.")

    rows = read_manifest(args.windows_manifest, args.split)
    if args.max_anchors > 0:
        rows = rows[: args.max_anchors]

    by_chrom: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_chrom[row["chrom"]].append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    fieldnames = ["anchor_id", "split", "source"] + [f"neg_id_{i}" for i in range(args.n_negatives)]
    source_counts: Counter[str] = Counter()
    short_rows = 0

    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for anchor in rows:
            negatives, source = choose_negatives(
                anchor=anchor,
                rows=rows,
                by_chrom=by_chrom,
                rng=rng,
                n_negatives=args.n_negatives,
                min_same_chrom_distance=args.min_same_chrom_distance,
            )
            if len(negatives) < args.n_negatives:
                short_rows += 1
                if not args.allow_fewer:
                    raise ValueError(
                        f"Anchor {anchor['id']} only has {len(negatives)} unique negatives; "
                        "pass --allow-fewer to write partial rows."
                    )
            out_row = {
                "anchor_id": anchor["id"],
                "split": anchor["split"],
                "source": source,
            }
            for i in range(args.n_negatives):
                out_row[f"neg_id_{i}"] = negatives[i]["id"] if i < len(negatives) else ""
            writer.writerow(out_row)
            source_counts[source] += 1

    print(f"Input manifest           : {args.windows_manifest}")
    print(f"Output negatives         : {output}")
    print(f"Split                    : {args.split}")
    print(f"Anchors written          : {len(rows):,}")
    print(f"Negatives per anchor     : {args.n_negatives}")
    print(f"Min same-chrom distance  : {args.min_same_chrom_distance:,} bp")
    print(f"Rows with fewer negatives: {short_rows:,}")
    print(f"Negative source counts   : {dict(source_counts)}")


if __name__ == "__main__":
    main()
