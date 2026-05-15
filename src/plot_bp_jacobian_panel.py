"""
Plot bp-mutagenesis contact maps with a shared color scale and optional junction.

Reads the .npz files written by bp_mutagenesis_jacobian_dna.py and writes:
  - one panel PNG/PDF
  - per-model redrawn PNGs
  - a CSV with enhancer/promoter block summary statistics when --junction is set
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


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
        items.append((label_from_path(path), path, data["contact"].astype(np.float32)))
    return items


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Input .npz files")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--junction", type=int, default=None, help="0-based matrix index for enhancer/promoter split")
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--symmetric", action="store_true", help="Use symmetric color limits around zero")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    import matplotlib.pyplot as plt

    items = load_contacts(args.inputs)
    vmin, vmax = shared_limits([x for _, _, x in items], args.percentile, args.symmetric)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)

    stats_rows = []
    for label, path, contact in items:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(contact, cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("mutable bp index")
        ax.set_ylabel("mutable bp index")
        if args.junction is not None:
            ax.axvline(args.junction - 0.5, color="black", lw=0.8, alpha=0.8)
            ax.axhline(args.junction - 0.5, color="black", lw=0.8, alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(f"{args.output_prefix}_{label}.png", dpi=args.dpi)
        plt.close(fig)

        if args.junction is not None:
            for block, vals in block_stats(contact, args.junction).items():
                stats_rows.append({
                    "label": label,
                    "path": str(path),
                    "block": block,
                    **vals,
                })

    n = len(items)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    last_im = None
    for ax, (label, _path, contact) in zip(axes, items):
        last_im = ax.imshow(contact, cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        if args.junction is not None:
            ax.axvline(args.junction - 0.5, color="black", lw=0.7, alpha=0.85)
            ax.axhline(args.junction - 0.5, color="black", lw=0.7, alpha=0.85)
    for ax in axes[n:]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:n].tolist(), fraction=0.025, pad=0.02)
    fig.savefig(f"{args.output_prefix}_panel.png", dpi=args.dpi)
    fig.savefig(f"{args.output_prefix}_panel.pdf")
    plt.close(fig)

    if stats_rows:
        with open(f"{args.output_prefix}_block_stats.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["label", "path", "block", "mean", "median", "p95"],
            )
            writer.writeheader()
            writer.writerows(stats_rows)

    print(f"Shared color limits: vmin={vmin:.4g}, vmax={vmax:.4g}")
    print(f"Wrote {args.output_prefix}_panel.png/.pdf and per-model PNGs")


if __name__ == "__main__":
    main()
