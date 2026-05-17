"""
Analyze bp-mutagenesis contact maps.

Reads .npz files from bp_mutagenesis_jacobian_dna.py and reports block-level
cross-vs-within metrics, long-range signal fractions, and pairwise model
similarity. Junctions can be supplied as bp coordinates and are mapped through
each file's retained positions array.
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


def load_npz(path: str):
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    if "contact" not in data.files:
        raise ValueError(f"{p} does not contain a contact array")
    positions = data["positions"].astype(np.int32) if "positions" in data.files else None
    return label_from_path(p), p, data["contact"].astype(np.float32), positions


def junction_index(positions: np.ndarray | None, junction_bp: int | None, junction: int | None):
    if junction_bp is None:
        return junction
    if positions is None:
        raise ValueError("--junction-bp requires .npz files with positions")
    return int(np.searchsorted(positions, junction_bp, side="left"))


def preprocess(contact: np.ndarray, mask_diagonal: int, mask_junction: int, junction: int | None):
    x = contact.astype(np.float32).copy()
    n = x.shape[0]
    if mask_diagonal > 0:
        idx = np.arange(n)
        x[np.abs(idx[:, None] - idx[None, :]) < mask_diagonal] = np.nan
    if mask_junction > 0 and junction is not None:
        lo = max(int(junction) - mask_junction, 0)
        hi = min(int(junction) + mask_junction + 1, n)
        x[lo:hi, :] = np.nan
        x[:, lo:hi] = np.nan
    return x


def finite_values(x: np.ndarray):
    return x[np.isfinite(x)]


def cohen_d(a: np.ndarray, b: np.ndarray):
    a = finite_values(a)
    b = finite_values(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    if pooled < 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def block_metrics(contact: np.ndarray, junction: int):
    j = min(max(int(junction), 0), contact.shape[0])
    cross = np.concatenate([contact[:j, j:].ravel(), contact[j:, :j].ravel()])
    within = np.concatenate([contact[:j, :j].ravel(), contact[j:, j:].ravel()])
    cross_abs = np.abs(cross)
    within_abs = np.abs(within)
    cross_f = finite_values(cross)
    within_f = finite_values(within)
    cross_abs_f = finite_values(cross_abs)
    within_abs_f = finite_values(within_abs)
    cross_p95 = float(np.nanpercentile(cross, 95))
    within_p95 = float(np.nanpercentile(within, 95))
    cross_abs_p95 = float(np.nanpercentile(cross_abs, 95))
    within_abs_p95 = float(np.nanpercentile(within_abs, 95))
    return {
        "cross_mean": float(np.nanmean(cross)),
        "within_mean": float(np.nanmean(within)),
        "cross_median": float(np.nanmedian(cross)),
        "within_median": float(np.nanmedian(within)),
        "cohen_d_signed": cohen_d(cross, within),
        "cohen_d_abs": cohen_d(cross_abs, within_abs),
        "cross_p95": cross_p95,
        "within_p95": within_p95,
        "p95_contrast": float((cross_p95 - within_p95) / max(abs(cross_p95) + abs(within_p95), 1e-12)),
        "cross_abs_p95": cross_abs_p95,
        "within_abs_p95": within_abs_p95,
        "abs_p95_contrast": float(
            (cross_abs_p95 - within_abs_p95) / max(abs(cross_abs_p95) + abs(within_abs_p95), 1e-12)
        ),
        "cross_n": int(cross_f.size),
        "within_n": int(within_f.size),
    }


def long_range_fraction(contact: np.ndarray, d_thresh: int):
    n = contact.shape[0]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    abs_c = np.abs(np.where(np.isfinite(contact), contact, 0.0))
    denom = float(abs_c.sum())
    if denom < 1e-12:
        return float("nan")
    return float(abs_c[sep > d_thresh].sum() / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray):
    try:
        from scipy.stats import spearmanr

        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return float("nan")
        return float(spearmanr(a[mask], b[mask]).correlation)
    except Exception:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return float("nan")
        ra = np.argsort(np.argsort(a[mask]))
        rb = np.argsort(np.argsort(b[mask]))
        return float(np.corrcoef(ra, rb)[0, 1])


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def corr_matrix_from_rows(labels: list[str], corr_rows: list[dict]):
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float32)
    for row in corr_rows:
        i = index[row["model_a"]]
        j = index[row["model_b"]]
        matrix[i, j] = row["spearman"]
    np.fill_diagonal(matrix, 1.0)
    return matrix


def cluster_order_from_corr(corr: np.ndarray):
    if corr.shape[0] < 2:
        return list(range(corr.shape[0])), None
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        clean = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        clean = np.clip((clean + clean.T) / 2.0, -1.0, 1.0)
        dist = 1.0 - clean
        np.fill_diagonal(dist, 0.0)
        linkage_matrix = linkage(squareform(dist, checks=False), method="average")
        return leaves_list(linkage_matrix).astype(int).tolist(), linkage_matrix
    except Exception as exc:
        print(f"WARNING: clustering unavailable ({exc}); using input order")
        return list(range(corr.shape[0])), None


def plot_spearman_heatmap(path: str, labels: list[str], corr: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(labels)
    fig_size = max(4.0, 0.55 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_dendrogram(path: str, labels: list[str], linkage_matrix):
    if linkage_matrix is None:
        return False
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    width = max(5.0, 0.55 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 4.0), constrained_layout=True)
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=45, ax=ax)
    ax.set_ylabel("1 - Spearman rho")
    ax.set_title("Contact-map similarity clustering")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--junction", type=int, default=None)
    p.add_argument("--junction-bp", type=int, default=None)
    p.add_argument("--mask-diagonal", type=int, default=5)
    p.add_argument("--mask-junction", type=int, default=0)
    p.add_argument("--long-range-thresholds", type=int, nargs="+", default=[100, 200, 500, 1000])
    args = p.parse_args()
    if args.junction is not None and args.junction_bp is not None:
        raise ValueError("Use either --junction or --junction-bp, not both")

    items = []
    rows = []
    for raw in args.inputs:
        label, path, contact, positions = load_npz(raw)
        j = junction_index(positions, args.junction_bp, args.junction)
        x = preprocess(contact, args.mask_diagonal, args.mask_junction, j)
        row = {
            "label": label,
            "path": str(path),
            "junction_bp": args.junction_bp,
            "junction_index": j,
            "n_positions": x.shape[0],
            "mask_diagonal": args.mask_diagonal,
            "mask_junction": args.mask_junction,
        }
        if j is not None:
            row.update(block_metrics(x, j))
        for thresh in args.long_range_thresholds:
            row[f"long_range_fraction_gt_{thresh}"] = long_range_fraction(x, thresh)
        rows.append(row)
        items.append((label, x, positions))

    metric_fields = list(rows[0].keys())
    write_csv(f"{args.output_prefix}_metrics.csv", rows, metric_fields)

    corr_rows = []
    for label_a, contact_a, positions_a in items:
        for label_b, contact_b, positions_b in items:
            if positions_a is not None and positions_b is not None:
                b_index = {int(pos): idx for idx, pos in enumerate(positions_b)}
                aligned = [
                    (idx_a, b_index[int(pos)], int(pos))
                    for idx_a, pos in enumerate(positions_a)
                    if int(pos) in b_index
                ]
                if len(aligned) < 2:
                    corr = float("nan")
                    n_aligned = len(aligned)
                    mode = "positions_intersection"
                else:
                    idx_a = np.array([item[0] for item in aligned], dtype=np.int32)
                    idx_b = np.array([item[1] for item in aligned], dtype=np.int32)
                    corr = spearman_corr(
                        contact_a[np.ix_(idx_a, idx_a)].ravel(),
                        contact_b[np.ix_(idx_b, idx_b)].ravel(),
                    )
                    n_aligned = len(aligned)
                    mode = "positions_intersection"
            else:
                n = min(contact_a.shape[0], contact_b.shape[0])
                corr = spearman_corr(contact_a[:n, :n].ravel(), contact_b[:n, :n].ravel())
                n_aligned = n
                mode = "matrix_prefix_fallback"
            corr_rows.append({
                "model_a": label_a,
                "model_b": label_b,
                "spearman": corr,
                "n_aligned_positions": n_aligned,
                "alignment_mode": mode,
            })
    write_csv(
        f"{args.output_prefix}_spearman.csv",
        corr_rows,
        ["model_a", "model_b", "spearman", "n_aligned_positions", "alignment_mode"],
    )

    labels = [item[0] for item in items]
    corr = corr_matrix_from_rows(labels, corr_rows)
    order, linkage_matrix = cluster_order_from_corr(corr)
    ordered_labels = [labels[i] for i in order]
    ordered_corr = corr[np.ix_(order, order)]
    write_csv(
        f"{args.output_prefix}_cluster_order.csv",
        [{"rank": rank + 1, "label": label} for rank, label in enumerate(ordered_labels)],
        ["rank", "label"],
    )
    plot_spearman_heatmap(
        f"{args.output_prefix}_spearman_heatmap.png",
        labels,
        corr,
        "Contact-map Spearman similarity",
    )
    plot_spearman_heatmap(
        f"{args.output_prefix}_spearman_clustered_heatmap.png",
        ordered_labels,
        ordered_corr,
        "Clustered contact-map Spearman similarity",
    )
    wrote_dendrogram = plot_dendrogram(
        f"{args.output_prefix}_spearman_dendrogram.png",
        labels,
        linkage_matrix,
    )

    print(f"Wrote {args.output_prefix}_metrics.csv")
    print(f"Wrote {args.output_prefix}_spearman.csv")
    print(f"Wrote {args.output_prefix}_cluster_order.csv")
    print(f"Wrote {args.output_prefix}_spearman_heatmap.png")
    print(f"Wrote {args.output_prefix}_spearman_clustered_heatmap.png")
    if wrote_dendrogram:
        print(f"Wrote {args.output_prefix}_spearman_dendrogram.png")


if __name__ == "__main__":
    main()
