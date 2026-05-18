"""
Plot bp-mutagenesis contact maps with a shared color scale and optional junction.

Reads the .npz files written by bp_mutagenesis_jacobian_dna.py and writes:
  - one panel PNG/PDF
  - per-model redrawn PNGs
  - a CSV with enhancer/promoter block summary statistics when a junction is set
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
GROUP_COLORS = {
    "promoter": "#e69f00",
    "tfbs": "#0072b2",
    "insulator": "#cc79a7",
    "motif": "#009e73",
}


def label_from_path(path: Path) -> str:
    name = path.stem
    for suffix in ("_gm12878_cj", "_cj", "_contact"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_contacts(paths: list[str]):
    items = []
    for raw in paths:
        path = Path(raw)
        data = np.load(path, allow_pickle=True)
        if "contact" not in data.files:
            raise ValueError(f"{path} does not contain a 'contact' array")
        positions = data["positions"].astype(np.int32) if "positions" in data.files else None
        sequence = str(np.asarray(data["sequence"]).item()).upper() if "sequence" in data.files else None
        contact_apc_applied = None
        if "contact_apc_applied" in data.files:
            contact_apc_applied = bool(np.asarray(data["contact_apc_applied"]).item())
        items.append((
            label_from_path(path),
            path,
            data["contact"].astype(np.float32),
            positions,
            sequence,
            contact_apc_applied,
        ))
    return items


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
                hits.append({
                    "motif": motif["name"],
                    "group": motif["group"],
                    "strand": strand,
                    "start": match.start(),
                    "end": match.start() + len(pattern),
                })
    return hits


def motif_track_codes(sequence: str | None, positions: np.ndarray | None, motifs: list[dict] | None, flank: int, include_rc: bool):
    if sequence is None or positions is None or motifs is None:
        return None, []
    groups = sorted({motif["group"] for motif in motifs})
    group_to_code = {group: idx + 1 for idx, group in enumerate(groups)}
    bp_codes = np.zeros(len(sequence), dtype=np.int16)
    for hit in motif_hits(sequence, motifs, include_rc=include_rc):
        code = group_to_code[hit["group"]]
        lo = max(0, int(hit["start"]) - flank)
        hi = min(len(sequence), int(hit["end"]) + flank)
        # Keep the first assigned group to make overlapping motif tracks stable.
        segment = bp_codes[lo:hi]
        segment[segment == 0] = code
    codes = np.zeros(positions.shape[0], dtype=np.int16)
    valid = positions < len(bp_codes)
    codes[valid] = bp_codes[positions[valid]]
    return codes, groups


def motif_cmap(groups: list[str]):
    from matplotlib.colors import ListedColormap

    colors = ["#f2f2f2"] + [GROUP_COLORS.get(group, "#666666") for group in groups]
    return ListedColormap(colors)


def add_motif_track(ax, codes: np.ndarray | None, groups: list[str]):
    if codes is None or not groups:
        return
    cmap = motif_cmap(groups)
    top = ax.inset_axes([0.0, 1.012, 1.0, 0.035])
    top.imshow(codes[np.newaxis, :], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(groups))
    top.set_axis_off()
    left = ax.inset_axes([-0.045, 0.0, 0.035, 1.0])
    left.imshow(codes[:, np.newaxis], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(groups))
    left.set_axis_off()


def shared_limits(contacts: list[np.ndarray], percentile: float, symmetric: bool):
    values = np.concatenate([x[np.isfinite(x)].ravel() for x in contacts])
    if symmetric:
        vmax = float(np.percentile(np.abs(values), percentile))
        return -vmax, vmax
    lo = float(np.percentile(values, 100.0 - percentile))
    hi = float(np.percentile(values, percentile))
    return lo, hi


def block_stats(contact: np.ndarray, junction: int):
    j = min(max(int(junction), 0), contact.shape[0])
    blocks = {
        "enh_enh": contact[:j, :j],
        "enh_prom": contact[:j, j:],
        "prom_enh": contact[j:, :j],
        "prom_prom": contact[j:, j:],
    }
    rows = {}
    for name, block in blocks.items():
        finite = block[np.isfinite(block)]
        if finite.size == 0:
            rows[name] = {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
        else:
            rows[name] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
            }
    return rows


def add_block_contrasts(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    enh_prom = rows.get("enh_prom", {})
    prom_enh = rows.get("prom_enh", {})
    enh_enh = rows.get("enh_enh", {})
    prom_prom = rows.get("prom_prom", {})
    cross_median = float(np.nanmean([enh_prom.get("median", np.nan), prom_enh.get("median", np.nan)]))
    within_median = float(np.nanmean([enh_enh.get("median", np.nan), prom_prom.get("median", np.nan)]))
    cross_p95 = float(np.nanmean([enh_prom.get("p95", np.nan), prom_enh.get("p95", np.nan)]))
    within_p95 = float(np.nanmean([enh_enh.get("p95", np.nan), prom_prom.get("p95", np.nan)]))
    for vals in rows.values():
        vals["cross_minus_within_median"] = cross_median - within_median
        vals["cross_over_within_median"] = cross_median / max(abs(within_median), 1e-6)
        vals["cross_within_p95_contrast"] = (
            (cross_p95 - within_p95) / max(abs(cross_p95) + abs(within_p95), 1e-6)
        )
    return rows


def junction_matrix_index(positions: np.ndarray | None, junction_bp: int | None, junction_index: int | None):
    if junction_bp is None:
        return junction_index
    if positions is None:
        raise ValueError("--junction-bp requires input .npz files with a 'positions' array")
    # positions are 0-based bp coordinates retained after token-visible filtering.
    # The matrix split is the number of retained positions strictly before the bp boundary.
    return int(np.searchsorted(positions, junction_bp, side="left"))


def preprocess_contact(contact: np.ndarray, mask_diagonal: int, mask_junction: int, junction: int | None, apc: bool):
    x = contact.astype(np.float32).copy()
    diag_mask = None
    if mask_diagonal > 0:
        idx = np.arange(x.shape[0])
        diag_mask = np.abs(idx[:, None] - idx[None, :]) < mask_diagonal
        x[diag_mask] = np.nan
    junction_slice = None
    if mask_junction > 0 and junction is not None:
        lo = max(int(junction) - mask_junction, 0)
        hi = min(int(junction) + mask_junction + 1, x.shape[0])
        junction_slice = slice(lo, hi)
        x[junction_slice, :] = np.nan
        x[:, junction_slice] = np.nan

    if apc:
        row_mean = np.nanmean(x, axis=1, keepdims=True)
        col_mean = np.nanmean(x, axis=0, keepdims=True)
        global_mean = np.nanmean(x)
        if np.isfinite(global_mean) and abs(float(global_mean)) > 1e-8:
            x = x - (row_mean @ col_mean) / global_mean
        else:
            x = x - row_mean - col_mean + global_mean
        if diag_mask is not None:
            x[diag_mask] = np.nan
        if junction_slice is not None:
            x[junction_slice, :] = np.nan
            x[:, junction_slice] = np.nan
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Input .npz files")
    p.add_argument("--output-prefix", required=True)
    p.add_argument(
        "--junction",
        type=int,
        default=None,
        help="0-based matrix index for enhancer/promoter split; kept for backwards compatibility",
    )
    p.add_argument(
        "--junction-bp",
        type=int,
        default=None,
        help="0-based bp boundary in the original cropped sequence; mapped through each .npz positions array",
    )
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument(
        "--no-symmetric",
        dest="symmetric",
        action="store_false",
        help="Use asymmetric percentile color limits instead of symmetric limits around zero",
    )
    p.set_defaults(symmetric=True)
    p.add_argument("--mask-diagonal", type=int, default=0, help="Mask |i-j| < N before plotting/statistics")
    p.add_argument(
        "--mask-junction",
        type=int,
        default=0,
        help="Mask rows/columns within this many matrix indices of the mapped junction",
    )
    p.add_argument("--apc", action="store_true", help="Apply average-product correction after optional diagonal masking")
    p.add_argument(
        "--force-apc",
        action="store_true",
        help="Allow --apc even when an input contact map is already marked APC-corrected",
    )
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--motifs", default=None, help="Optional motif CSV with name,pattern,group columns")
    p.add_argument("--motif-flank", type=int, default=2, help="Expand motif hits by this many bp for plotting tracks")
    p.add_argument("--no-rc", action="store_true", help="Do not scan reverse-complement motif orientation")
    args = p.parse_args()
    if args.junction is not None and args.junction_bp is not None:
        raise ValueError("Use either --junction or --junction-bp, not both")

    import matplotlib.pyplot as plt

    motifs = read_motifs(args.motifs) if args.motifs else None
    raw_items = load_contacts(args.inputs)
    if args.apc:
        already_apc = [str(path) for _label, path, _contact, _positions, _sequence, apc_done in raw_items if apc_done is True]
        unknown_apc = [str(path) for _label, path, _contact, _positions, _sequence, apc_done in raw_items if apc_done is None]
        if already_apc and not args.force_apc:
            raise ValueError(
                "--apc would apply APC a second time to contact maps already marked as APC-corrected: "
                + ", ".join(already_apc)
                + ". Re-run without --apc, or pass --force-apc if this is intentional."
            )
        if unknown_apc:
            print(
                "WARNING: --apc requested for inputs without contact_apc_applied metadata; "
                "old bp-mutagenesis outputs may already be APC-corrected:",
                ", ".join(unknown_apc),
            )
    items = [
        (
            label,
            path,
            positions,
            sequence,
            junction_matrix_index(positions, args.junction_bp, args.junction),
            contact_apc_applied,
            contact,
        )
        for label, path, contact, positions, sequence, contact_apc_applied in raw_items
    ]
    items = [
        (
            label,
            path,
            preprocess_contact(contact, args.mask_diagonal, args.mask_junction, junction, args.apc),
            positions,
            sequence,
            junction,
            contact_apc_applied,
            *motif_track_codes(sequence, positions, motifs, args.motif_flank, include_rc=not args.no_rc),
        )
        for label, path, positions, sequence, junction, contact_apc_applied, contact in items
    ]
    for label, path, contact, _positions, _sequence, junction, _contact_apc_applied, _motif_codes, _motif_groups in items:
        if junction is not None:
            print(
                "Junction mapping:",
                label,
                f"bp={args.junction_bp if args.junction_bp is not None else 'n/a'}",
                f"index={junction}/{contact.shape[0]}",
                f"path={path}",
            )
    vmin, vmax = shared_limits([x for _, _, x, _, _, _, _, _, _ in items], args.percentile, args.symmetric)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)

    stats_rows = []
    for label, path, contact, positions, _sequence, junction, contact_apc_applied, motif_codes, motif_groups in items:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(contact, cmap=args.cmap, vmin=vmin, vmax=vmax)
        add_motif_track(ax, motif_codes, motif_groups)
        ax.set_title(label)
        ax.set_xlabel("mutable bp index")
        ax.set_ylabel("mutable bp index")
        if junction is not None:
            ax.axvline(junction - 0.5, color="black", lw=0.8, alpha=0.8)
            ax.axhline(junction - 0.5, color="black", lw=0.8, alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(f"{args.output_prefix}_{label}.png", dpi=args.dpi)
        plt.close(fig)

        if junction is not None:
            for block, vals in add_block_contrasts(block_stats(contact, junction)).items():
                stats_rows.append({
                    "label": label,
                    "path": str(path),
                    "junction_bp": args.junction_bp,
                    "junction_index": junction,
                    "n_positions": contact.shape[0],
                    "first_bp": None if positions is None or positions.size == 0 else int(positions[0]),
                    "last_bp": None if positions is None or positions.size == 0 else int(positions[-1]),
                    "input_contact_apc_applied": contact_apc_applied,
                    "plot_apc_applied": args.apc,
                    "mask_diagonal": args.mask_diagonal,
                    "mask_junction": args.mask_junction,
                    "block": block,
                    **vals,
                })

    n = len(items)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    last_im = None
    for ax, (label, _path, contact, _positions, _sequence, junction, _contact_apc_applied, motif_codes, motif_groups) in zip(axes, items):
        last_im = ax.imshow(contact, cmap=args.cmap, vmin=vmin, vmax=vmax)
        add_motif_track(ax, motif_codes, motif_groups)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        if junction is not None:
            ax.axvline(junction - 0.5, color="black", lw=0.7, alpha=0.85)
            ax.axhline(junction - 0.5, color="black", lw=0.7, alpha=0.85)
    for ax in axes[n:]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:n].tolist(), fraction=0.025, pad=0.02)
    if motifs:
        from matplotlib.patches import Patch

        all_groups = sorted({group for item in items for group in item[-1]})
        handles = [Patch(facecolor=GROUP_COLORS.get(group, "#666666"), label=group) for group in all_groups]
        if handles:
            fig.legend(handles=handles, loc="upper right", frameon=False, title="Motif group")
    fig.savefig(f"{args.output_prefix}_panel.png", dpi=args.dpi)
    fig.savefig(f"{args.output_prefix}_panel.pdf")
    plt.close(fig)

    if stats_rows:
        with open(f"{args.output_prefix}_block_stats.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "label",
                    "path",
                    "junction_bp",
                    "junction_index",
                    "n_positions",
                    "first_bp",
                    "last_bp",
                    "input_contact_apc_applied",
                    "plot_apc_applied",
                    "mask_diagonal",
                    "mask_junction",
                    "block",
                    "mean",
                    "median",
                    "p95",
                    "cross_minus_within_median",
                    "cross_over_within_median",
                    "cross_within_p95_contrast",
                ],
            )
            writer.writeheader()
            writer.writerows(stats_rows)

    print(f"Shared color limits: vmin={vmin:.4g}, vmax={vmax:.4g}")
    print(f"Wrote {args.output_prefix}_panel.png/.pdf and per-model PNGs")


if __name__ == "__main__":
    main()
