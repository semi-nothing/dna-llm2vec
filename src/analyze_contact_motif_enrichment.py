"""
Test whether bp-mutagenesis contact-map hotspots are enriched for sequence motifs.

The script reads .npz files written by bp_mutagenesis_jacobian_dna.py. It scans
the stored sequence with IUPAC motifs, maps motif-covered bp positions onto the
retained contact-matrix positions, and compares motif-pair frequency in top
contact pairs against a distance-matched background.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np


IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}
COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def label_from_path(path: Path) -> str:
    name = path.stem
    for suffix in ("_gm12878_cj", "_cj", "_contact"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def parse_contact_label(label: str):
    variant = re.match(r"^(m\d+)", label)
    epi = re.search(r"_([01])_(\d+)$", label)
    return {
        "model_variant": variant.group(1) if variant else "",
        "epi_label": epi.group(1) if epi else "",
        "sample_index": epi.group(2) if epi else "",
    }


def load_npz(path: str):
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    for key in ("contact", "positions", "sequence"):
        if key not in data.files:
            raise ValueError(f"{p} does not contain required array {key!r}")
    sequence = np.asarray(data["sequence"]).item()
    return label_from_path(p), p, data["contact"].astype(np.float32), data["positions"].astype(np.int32), str(sequence).upper()


def read_motifs(path: str):
    motifs = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "pattern"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            name = row["name"].strip()
            pattern = row["pattern"].strip().upper()
            group = row.get("group", "").strip() or "motif"
            if not name or not pattern:
                continue
            unknown = sorted(set(pattern) - set(IUPAC))
            if unknown:
                raise ValueError(f"Motif {name} contains unsupported IUPAC symbols: {unknown}")
            motifs.append({"name": name, "pattern": pattern, "group": group})
    if not motifs:
        raise ValueError(f"No motifs found in {path}")
    return motifs


def iupac_to_regex(pattern: str):
    return "".join(f"[{IUPAC[ch]}]" for ch in pattern.upper())


def reverse_complement_pattern(pattern: str):
    return pattern.translate(COMPLEMENT)[::-1].upper()


def motif_hits(sequence: str, motifs: list[dict], include_rc: bool):
    hits = []
    for motif in motifs:
        patterns = [(motif["pattern"], "+")]
        rc = reverse_complement_pattern(motif["pattern"])
        if include_rc and rc != motif["pattern"]:
            patterns.append((rc, "-"))
        for pattern, strand in patterns:
            regex = re.compile(f"(?=({iupac_to_regex(pattern)}))", re.IGNORECASE)
            for match in regex.finditer(sequence):
                start = match.start()
                end = start + len(pattern)
                hits.append({
                    "motif": motif["name"],
                    "group": motif["group"],
                    "pattern": motif["pattern"],
                    "strand": strand,
                    "start": start,
                    "end": end,
                })
    return hits


def build_masks(positions: np.ndarray, hits: list[dict], flank: int):
    covered_bp = np.zeros(int(positions.max()) + 1 if positions.size else 0, dtype=bool)
    motif_by_bp: list[set[str]] = [set() for _ in range(covered_bp.size)]
    for hit in hits:
        lo = max(0, int(hit["start"]) - flank)
        hi = min(covered_bp.size, int(hit["end"]) + flank)
        if lo >= hi:
            continue
        covered_bp[lo:hi] = True
        for bp in range(lo, hi):
            motif_by_bp[bp].add(hit["motif"])
    valid = positions < covered_bp.size
    mask = np.zeros(positions.shape[0], dtype=bool)
    mask[valid] = covered_bp[positions[valid]]
    names = [sorted(motif_by_bp[int(pos)]) if int(pos) < len(motif_by_bp) else [] for pos in positions]
    return mask, names


def pair_indices(n: int, min_sep: int):
    i, j = np.triu_indices(n, k=1)
    if min_sep > 1:
        keep = (j - i) >= min_sep
        i = i[keep]
        j = j[keep]
    return i.astype(np.int32), j.astype(np.int32)


def top_pair_mask(values: np.ndarray, top_fraction: float, top_k: int | None):
    finite = np.isfinite(values)
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    if top_k is None:
        top_k = max(1, int(round(finite_idx.size * top_fraction)))
    top_k = min(max(1, top_k), finite_idx.size)
    finite_values = values[finite_idx]
    chosen_local = np.argpartition(finite_values, -top_k)[-top_k:]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[finite_idx[chosen_local]] = True
    return mask


def distance_bins(distances: np.ndarray, edges: list[int]):
    edges_arr = np.array(sorted(set(edges)), dtype=np.int32)
    return np.searchsorted(edges_arr, distances, side="right")


def mean_or_nan(values: np.ndarray):
    return float(np.mean(values)) if values.size else float("nan")


def permutation_background(pair_is_motif: np.ndarray, top_mask: np.ndarray, bins: np.ndarray, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    bg = []
    for _ in range(n_perm):
        chosen = []
        for bin_id in np.unique(bins[top_mask]):
            n_take = int(np.sum(top_mask & (bins == bin_id)))
            pool = np.flatnonzero((~top_mask) & (bins == bin_id))
            if pool.size == 0 or n_take == 0:
                continue
            replace = pool.size < n_take
            chosen.append(rng.choice(pool, size=n_take, replace=replace))
        if chosen:
            idx = np.concatenate(chosen)
            bg.append(mean_or_nan(pair_is_motif[idx].astype(np.float32)))
    return np.array(bg, dtype=np.float32)


def summarize_one(path: str, motifs: list[dict], args):
    label, npz_path, contact, positions, sequence = load_npz(path)
    hits = motif_hits(sequence, motifs, include_rc=not args.no_rc)
    motif_mask, motif_names_by_pos = build_masks(positions, hits, args.motif_flank)

    x = contact.copy()
    if args.score == "abs":
        x = np.abs(x)
    elif args.score == "positive":
        x = np.where(x > 0, x, np.nan)
    elif args.score == "negative_abs":
        x = np.where(x < 0, np.abs(x), np.nan)
    else:
        raise ValueError(f"Unknown score mode: {args.score}")

    i, j = pair_indices(x.shape[0], args.min_sep)
    values = x[i, j]
    top_mask = top_pair_mask(values, args.top_fraction, args.top_k)
    pair_is_motif = motif_mask[i] & motif_mask[j]
    pair_one_motif = motif_mask[i] | motif_mask[j]
    bins = distance_bins(j - i, args.distance_bins)
    bg = permutation_background(pair_is_motif, top_mask, bins, args.n_perm, args.seed)

    top_fraction = mean_or_nan(pair_is_motif[top_mask].astype(np.float32))
    top_one_fraction = mean_or_nan(pair_one_motif[top_mask].astype(np.float32))
    all_fraction = mean_or_nan(pair_is_motif[np.isfinite(values)].astype(np.float32))
    bg_mean = float(np.mean(bg)) if bg.size else float("nan")
    bg_std = float(np.std(bg)) if bg.size else float("nan")
    z = float((top_fraction - bg_mean) / bg_std) if bg_std > 1e-12 else float("nan")
    p_emp = float((np.sum(bg >= top_fraction) + 1) / (bg.size + 1)) if bg.size else float("nan")
    enrichment = float(top_fraction / bg_mean) if bg_mean > 1e-12 else float("nan")

    summary = {
        "label": label,
        **parse_contact_label(label),
        "path": str(npz_path),
        "n_positions": int(positions.size),
        "sequence_length": len(sequence),
        "n_motifs": len(motifs),
        "n_motif_hits": len(hits),
        "motif_position_fraction": float(np.mean(motif_mask)) if motif_mask.size else float("nan"),
        "score": args.score,
        "top_fraction": args.top_fraction,
        "top_k": int(np.sum(top_mask)),
        "min_sep": args.min_sep,
        "motif_flank": args.motif_flank,
        "top_motif_pair_fraction": top_fraction,
        "top_one_endpoint_motif_fraction": top_one_fraction,
        "all_motif_pair_fraction": all_fraction,
        "distance_matched_background_mean": bg_mean,
        "distance_matched_background_std": bg_std,
        "motif_pair_enrichment": enrichment,
        "motif_pair_z": z,
        "motif_pair_empirical_p": p_emp,
    }

    hit_rows = []
    for hit in hits:
        hit_rows.append({"label": label, **hit})

    top_rows = []
    top_idx = np.flatnonzero(top_mask)
    top_idx = top_idx[np.argsort(values[top_idx])[::-1]]
    for rank, pair_idx in enumerate(top_idx[: args.write_top_pairs], start=1):
        a = int(i[pair_idx])
        b = int(j[pair_idx])
        top_rows.append({
            "label": label,
            "rank": rank,
            "i": a + 1,
            "j": b + 1,
            "i_bp": int(positions[a]) + 1,
            "j_bp": int(positions[b]) + 1,
            "distance": int(positions[b] - positions[a]),
            "score": float(values[pair_idx]),
            "i_motifs": ";".join(motif_names_by_pos[a]),
            "j_motifs": ";".join(motif_names_by_pos[b]),
            "motif_pair": bool(pair_is_motif[pair_idx]),
        })
    return summary, hit_rows, top_rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Input contact-map .npz files")
    p.add_argument("--motifs", default="data/motif_panels/core_regulatory_iupac.csv")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--score", choices=("abs", "positive", "negative_abs"), default="abs")
    p.add_argument("--top-fraction", type=float, default=0.01)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-sep", type=int, default=50, help="Minimum matrix-index separation for evaluated pairs")
    p.add_argument("--motif-flank", type=int, default=0, help="Expand motif hits by this many bp on both sides")
    p.add_argument("--distance-bins", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-rc", action="store_true", help="Do not scan reverse-complement motif orientation")
    p.add_argument("--write-top-pairs", type=int, default=200)
    args = p.parse_args()
    if args.top_k is None and not (0 < args.top_fraction <= 1):
        raise ValueError("--top-fraction must be in (0, 1] when --top-k is not set")

    motifs = read_motifs(args.motifs)
    summary_rows = []
    hit_rows = []
    top_pair_rows = []
    for raw in args.inputs:
        summary, hits, top_pairs = summarize_one(raw, motifs, args)
        summary_rows.append(summary)
        hit_rows.extend(hits)
        top_pair_rows.extend(top_pairs)

    write_csv(f"{args.output_prefix}_summary.csv", summary_rows)
    write_csv(
        f"{args.output_prefix}_motif_hits.csv",
        hit_rows,
        ["label", "motif", "group", "pattern", "strand", "start", "end"],
    )
    write_csv(
        f"{args.output_prefix}_top_pairs.csv",
        top_pair_rows,
        ["label", "rank", "i", "j", "i_bp", "j_bp", "distance", "score", "i_motifs", "j_motifs", "motif_pair"],
    )
    print(f"Wrote {args.output_prefix}_summary.csv")
    print(f"Wrote {args.output_prefix}_motif_hits.csv")
    print(f"Wrote {args.output_prefix}_top_pairs.csv")


if __name__ == "__main__":
    main()
