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
import math
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
    variant = re.match(r"^(m\d+)", label, flags=re.IGNORECASE)
    epi = re.search(r"(?:^|_)gm12878_([01])_(\d+)$", label, flags=re.IGNORECASE)
    hg38 = re.search(r"_(chr[^_]+)_(\d+)_(\d+)$", label, flags=re.IGNORECASE)
    parsed = {
        "model_variant": variant.group(1) if variant else "",
        "epi_label": epi.group(1) if epi else "",
        "sample_index": epi.group(2) if epi else "",
        "chrom": "",
        "window_start_1based": "",
        "window_end_1based": "",
    }
    if hg38:
        parsed.update({
            "chrom": hg38.group(1),
            "window_start_1based": hg38.group(2),
            "window_end_1based": hg38.group(3),
        })
    return parsed


def run_self_test():
    cases = {
        "m0_dnagpt_gm12878_0_3": {
            "model_variant": "m0",
            "epi_label": "0",
            "sample_index": "3",
            "chrom": "",
        },
        "m4_r1_bidir_gm12878_1_12": {
            "model_variant": "m4",
            "epi_label": "1",
            "sample_index": "12",
            "chrom": "",
        },
        "m0_dnagpt_chr3_100_4096": {
            "model_variant": "m0",
            "epi_label": "",
            "sample_index": "",
            "chrom": "chr3",
            "window_start_1based": "100",
            "window_end_1based": "4096",
        },
    }
    for label, expected in cases.items():
        parsed = parse_contact_label(label)
        for key, value in expected.items():
            if parsed[key] != value:
                raise AssertionError(f"{label}: expected {key}={value!r}, got {parsed[key]!r}")
    print("Self-test passed")


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


def build_masks(positions: np.ndarray, hits: list[dict], flank: int, motif_to_bit: dict[str, int], group_to_bit: dict[str, int]):
    covered_bp = np.zeros(int(positions.max()) + 1 if positions.size else 0, dtype=bool)
    motif_by_bp: list[set[str]] = [set() for _ in range(covered_bp.size)]
    group_by_bp: list[set[str]] = [set() for _ in range(covered_bp.size)]
    motif_bits_by_bp = np.zeros(covered_bp.size, dtype=np.uint64)
    group_bits_by_bp = np.zeros(covered_bp.size, dtype=np.uint64)
    for hit in hits:
        lo = max(0, int(hit["start"]) - flank)
        hi = min(covered_bp.size, int(hit["end"]) + flank)
        if lo >= hi:
            continue
        covered_bp[lo:hi] = True
        motif_bit = np.uint64(motif_to_bit[hit["motif"]])
        group_bit = np.uint64(group_to_bit[hit["group"]])
        motif_bits_by_bp[lo:hi] |= motif_bit
        group_bits_by_bp[lo:hi] |= group_bit
        for bp in range(lo, hi):
            motif_by_bp[bp].add(hit["motif"])
            group_by_bp[bp].add(hit["group"])
    valid = positions < covered_bp.size
    mask = np.zeros(positions.shape[0], dtype=bool)
    mask[valid] = covered_bp[positions[valid]]
    names = [sorted(motif_by_bp[int(pos)]) if int(pos) < len(motif_by_bp) else [] for pos in positions]
    groups = [sorted(group_by_bp[int(pos)]) if int(pos) < len(group_by_bp) else [] for pos in positions]
    motif_bits = np.zeros(positions.shape[0], dtype=np.uint64)
    group_bits = np.zeros(positions.shape[0], dtype=np.uint64)
    motif_bits[valid] = motif_bits_by_bp[positions[valid]]
    group_bits[valid] = group_bits_by_bp[positions[valid]]
    return mask, names, groups, motif_bits, group_bits


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


def enrichment_stats(pair_mask: np.ndarray, top_mask: np.ndarray, finite_mask: np.ndarray, bins: np.ndarray, args, seed_offset: int = 0):
    bg = permutation_background(pair_mask, top_mask, bins, args.n_perm, args.seed + seed_offset)
    top_fraction = mean_or_nan(pair_mask[top_mask].astype(np.float32))
    all_fraction = mean_or_nan(pair_mask[finite_mask].astype(np.float32))
    bg_mean = float(np.mean(bg)) if bg.size else float("nan")
    bg_std = float(np.std(bg)) if bg.size else float("nan")
    z = float((top_fraction - bg_mean) / bg_std) if bg_std > 1e-12 else float("nan")
    p_emp = float((np.sum(bg >= top_fraction) + 1) / (bg.size + 1)) if bg.size else float("nan")
    enrichment = float(top_fraction / bg_mean) if bg_mean > 1e-12 else float("nan")
    return {
        "top_fraction": top_fraction,
        "all_fraction": all_fraction,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "enrichment": enrichment,
        "z": z,
        "empirical_p": p_emp,
    }


def summarize_one(path: str, motifs: list[dict], args):
    label, npz_path, contact, positions, sequence = load_npz(path)
    hits = motif_hits(sequence, motifs, include_rc=not args.no_rc)
    motif_to_bit = {motif["name"]: 1 << idx for idx, motif in enumerate(motifs)}
    group_to_bit = {group: 1 << idx for idx, group in enumerate(sorted({motif["group"] for motif in motifs}))}
    if len(motif_to_bit) > 63 or len(group_to_bit) > 63:
        raise ValueError("Bitset motif/group matching supports at most 63 motifs and 63 groups")
    (
        motif_mask,
        motif_names_by_pos,
        motif_groups_by_pos,
        motif_bits_by_pos,
        group_bits_by_pos,
    ) = build_masks(positions, hits, args.motif_flank, motif_to_bit, group_to_bit)

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
    finite_mask = np.isfinite(values)
    pair_is_motif = motif_mask[i] & motif_mask[j]
    pair_one_motif = motif_mask[i] | motif_mask[j]
    pair_same_motif = (motif_bits_by_pos[i] & motif_bits_by_pos[j]) != 0
    bins = distance_bins(j - i, args.distance_bins)
    motif_pair_stats = enrichment_stats(pair_is_motif, top_mask, finite_mask, bins, args, seed_offset=0)
    same_motif_stats = enrichment_stats(pair_same_motif, top_mask, finite_mask, bins, args, seed_offset=1009)
    top_one_fraction = mean_or_nan(pair_one_motif[top_mask].astype(np.float32))

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
        "top_motif_pair_fraction": motif_pair_stats["top_fraction"],
        "top_one_endpoint_motif_fraction": top_one_fraction,
        "all_motif_pair_fraction": motif_pair_stats["all_fraction"],
        "distance_matched_background_mean": motif_pair_stats["background_mean"],
        "distance_matched_background_std": motif_pair_stats["background_std"],
        "motif_pair_enrichment": motif_pair_stats["enrichment"],
        "motif_pair_z": motif_pair_stats["z"],
        "motif_pair_empirical_p": motif_pair_stats["empirical_p"],
        "top_same_motif_pair_fraction": same_motif_stats["top_fraction"],
        "all_same_motif_pair_fraction": same_motif_stats["all_fraction"],
        "same_motif_background_mean": same_motif_stats["background_mean"],
        "same_motif_pair_enrichment": same_motif_stats["enrichment"],
        "same_motif_pair_z": same_motif_stats["z"],
        "same_motif_pair_empirical_p": same_motif_stats["empirical_p"],
    }

    group_rows = []
    if args.per_motif_group:
        groups = sorted({motif["group"] for motif in motifs})
        for group_idx, group in enumerate(groups):
            group_bit = np.uint64(group_to_bit[group])
            group_mask = (group_bits_by_pos & group_bit) != 0
            pair_is_group = group_mask[i] & group_mask[j]
            group_same_motif = pair_same_motif & pair_is_group
            group_pair_stats = enrichment_stats(
                pair_is_group, top_mask, finite_mask, bins, args, seed_offset=2000 + group_idx
            )
            group_same_stats = enrichment_stats(
                group_same_motif, top_mask, finite_mask, bins, args, seed_offset=3000 + group_idx
            )
            group_rows.append({
                "label": label,
                **parse_contact_label(label),
                "group": group,
                "group_position_fraction": float(np.mean(group_mask)) if group_mask.size else float("nan"),
                "top_group_pair_fraction": group_pair_stats["top_fraction"],
                "all_group_pair_fraction": group_pair_stats["all_fraction"],
                "group_background_mean": group_pair_stats["background_mean"],
                "group_pair_enrichment": group_pair_stats["enrichment"],
                "group_pair_z": group_pair_stats["z"],
                "group_pair_empirical_p": group_pair_stats["empirical_p"],
                "top_group_same_motif_fraction": group_same_stats["top_fraction"],
                "all_group_same_motif_fraction": group_same_stats["all_fraction"],
                "group_same_motif_background_mean": group_same_stats["background_mean"],
                "group_same_motif_enrichment": group_same_stats["enrichment"],
                "group_same_motif_z": group_same_stats["z"],
                "group_same_motif_empirical_p": group_same_stats["empirical_p"],
            })

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
            "i_groups": ";".join(motif_groups_by_pos[a]),
            "j_groups": ";".join(motif_groups_by_pos[b]),
            "motif_pair": bool(pair_is_motif[pair_idx]),
            "same_motif_pair": bool(pair_same_motif[pair_idx]),
        })
    return summary, hit_rows, top_rows, group_rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normal_sf(z: float):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def combine_stouffer(z_values: list[float]):
    vals = [z for z in z_values if np.isfinite(z)]
    if not vals:
        return float("nan"), float("nan")
    z = float(np.sum(vals) / math.sqrt(len(vals)))
    return z, normal_sf(z)


def combine_fisher(p_values: list[float]):
    vals = [p for p in p_values if np.isfinite(p) and p > 0]
    if not vals:
        return float("nan"), float("nan")
    stat = float(-2.0 * np.sum(np.log(vals)))
    try:
        from scipy.stats import chi2

        p = float(chi2.sf(stat, 2 * len(vals)))
    except Exception:
        p = float("nan")
    return stat, p


def aggregate_rows(rows: list[dict], keys: list[str]):
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(item, "") for item in keys)
        groups.setdefault(key, []).append(row)

    out = []
    numeric_fields = [
        "motif_position_fraction",
        "top_motif_pair_fraction",
        "top_one_endpoint_motif_fraction",
        "all_motif_pair_fraction",
        "distance_matched_background_mean",
        "motif_pair_enrichment",
        "motif_pair_z",
        "motif_pair_empirical_p",
        "top_same_motif_pair_fraction",
        "all_same_motif_pair_fraction",
        "same_motif_background_mean",
        "same_motif_pair_enrichment",
        "same_motif_pair_z",
        "same_motif_pair_empirical_p",
    ]
    for key, group in sorted(groups.items()):
        item = {name: value for name, value in zip(keys, key)}
        item["n"] = len(group)
        for field in numeric_fields:
            vals = [float(row[field]) for row in group if row.get(field, "") not in ("", None)]
            vals = [val for val in vals if np.isfinite(val)]
            item[f"mean_{field}"] = float(np.mean(vals)) if vals else float("nan")
            item[f"median_{field}"] = float(np.median(vals)) if vals else float("nan")
        stouffer_z, stouffer_p = combine_stouffer([float(row["motif_pair_z"]) for row in group])
        fisher_stat, fisher_p = combine_fisher([float(row["motif_pair_empirical_p"]) for row in group])
        item["stouffer_z"] = stouffer_z
        item["stouffer_p_one_sided"] = stouffer_p
        item["fisher_empirical_p_chi2"] = fisher_stat
        item["fisher_empirical_p"] = fisher_p
        out.append(item)
    return out


def aggregate_group_rows(rows: list[dict], keys: list[str]):
    group_keys = keys + ["group"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(item, "") for item in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    numeric_fields = [
        "group_position_fraction",
        "top_group_pair_fraction",
        "all_group_pair_fraction",
        "group_background_mean",
        "group_pair_enrichment",
        "group_pair_z",
        "group_pair_empirical_p",
        "top_group_same_motif_fraction",
        "all_group_same_motif_fraction",
        "group_same_motif_background_mean",
        "group_same_motif_enrichment",
        "group_same_motif_z",
        "group_same_motif_empirical_p",
    ]
    for key, group in sorted(groups.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        item["n"] = len(group)
        for field in numeric_fields:
            vals = [float(row[field]) for row in group if row.get(field, "") not in ("", None)]
            vals = [val for val in vals if np.isfinite(val)]
            item[f"mean_{field}"] = float(np.mean(vals)) if vals else float("nan")
            item[f"median_{field}"] = float(np.median(vals)) if vals else float("nan")
        stouffer_z, stouffer_p = combine_stouffer([float(row["group_pair_z"]) for row in group])
        fisher_stat, fisher_p = combine_fisher([float(row["group_pair_empirical_p"]) for row in group])
        same_stouffer_z, same_stouffer_p = combine_stouffer([float(row["group_same_motif_z"]) for row in group])
        same_fisher_stat, same_fisher_p = combine_fisher([float(row["group_same_motif_empirical_p"]) for row in group])
        item["group_stouffer_z"] = stouffer_z
        item["group_stouffer_p_one_sided"] = stouffer_p
        item["group_fisher_empirical_p_chi2"] = fisher_stat
        item["group_fisher_empirical_p"] = fisher_p
        item["group_same_motif_stouffer_z"] = same_stouffer_z
        item["group_same_motif_stouffer_p_one_sided"] = same_stouffer_p
        item["group_same_motif_fisher_empirical_p_chi2"] = same_fisher_stat
        item["group_same_motif_fisher_empirical_p"] = same_fisher_p
        out.append(item)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", default=None, help="Input contact-map .npz files")
    p.add_argument("--motifs", default="data/motif_panels/core_regulatory_iupac.csv")
    p.add_argument("--output-prefix", required=True)
    p.add_argument(
        "--score",
        choices=("abs", "positive", "negative_abs"),
        default="positive",
        help=(
            "Contact score used to define top pairs. 'positive' is the default for motif co-occurrence, "
            "where same-direction perturbation effects are the primary signal. Use 'abs' as a sensitivity check."
        ),
    )
    p.add_argument("--top-fraction", type=float, default=0.01)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-sep", type=int, default=50, help="Minimum matrix-index separation for evaluated pairs")
    p.add_argument(
        "--motif-flank",
        type=int,
        default=2,
        help="Expand motif hits by this many bp on both sides to tolerate token/motif boundary offsets",
    )
    p.add_argument("--distance-bins", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
    p.add_argument(
        "--n-perm",
        type=int,
        default=200,
        help=(
            "Distance-matched permutations. Use >=1000 for paper-level per-window p-values; "
            "use >=5000 if interpreting Fisher-combined empirical p-values."
        ),
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-rc", action="store_true", help="Do not scan reverse-complement motif orientation")
    p.add_argument("--write-top-pairs", type=int, default=200)
    p.add_argument("--per-motif-group", action="store_true", help="Write motif-group-specific enrichment results")
    p.add_argument(
        "--aggregate-by",
        nargs="+",
        default=[],
        choices=("model_variant", "epi_label", "sample_index", "chrom"),
        help="Optional columns used to write an aggregate summary CSV, e.g. --aggregate-by model_variant",
    )
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.inputs:
        raise ValueError("--inputs is required unless --self-test is used")
    if args.top_k is None and not (0 < args.top_fraction <= 1):
        raise ValueError("--top-fraction must be in (0, 1] when --top-k is not set")

    motifs = read_motifs(args.motifs)
    summary_rows = []
    hit_rows = []
    top_pair_rows = []
    group_rows = []
    for raw in args.inputs:
        summary, hits, top_pairs, groups = summarize_one(raw, motifs, args)
        summary_rows.append(summary)
        hit_rows.extend(hits)
        top_pair_rows.extend(top_pairs)
        group_rows.extend(groups)

    write_csv(f"{args.output_prefix}_summary.csv", summary_rows)
    write_csv(
        f"{args.output_prefix}_motif_hits.csv",
        hit_rows,
        ["label", "motif", "group", "pattern", "strand", "start", "end"],
    )
    write_csv(
        f"{args.output_prefix}_top_pairs.csv",
        top_pair_rows,
        [
            "label",
            "rank",
            "i",
            "j",
            "i_bp",
            "j_bp",
            "distance",
            "score",
            "i_motifs",
            "j_motifs",
            "i_groups",
            "j_groups",
            "motif_pair",
            "same_motif_pair",
        ],
    )
    if args.per_motif_group:
        write_csv(f"{args.output_prefix}_motif_group_summary.csv", group_rows)
    if args.aggregate_by:
        aggregate = aggregate_rows(summary_rows, args.aggregate_by)
        aggregate_suffix = "_".join(args.aggregate_by)
        write_csv(f"{args.output_prefix}_aggregate_{aggregate_suffix}.csv", aggregate)
        if group_rows:
            group_aggregate = aggregate_group_rows(group_rows, args.aggregate_by)
            write_csv(f"{args.output_prefix}_aggregate_groups_{aggregate_suffix}.csv", group_aggregate)
    print(f"Wrote {args.output_prefix}_summary.csv")
    print(f"Wrote {args.output_prefix}_motif_hits.csv")
    print(f"Wrote {args.output_prefix}_top_pairs.csv")
    if args.per_motif_group:
        print(f"Wrote {args.output_prefix}_motif_group_summary.csv")
    if args.aggregate_by:
        print(f"Wrote {args.output_prefix}_aggregate_{aggregate_suffix}.csv")
        if group_rows:
            print(f"Wrote {args.output_prefix}_aggregate_groups_{aggregate_suffix}.csv")


if __name__ == "__main__":
    main()
