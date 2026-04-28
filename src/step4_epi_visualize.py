"""
Step 4 add-on: EPI embedding visualisation for paper figures.

This script reuses the Step 4 loading / encoding pipeline, but only for the
GUE+ EPI dev split. It extracts sequence embeddings, projects them to 2D with
t-SNE / UMAP / PCA, and saves:

  - a publication-friendly scatter plot
  - per-model summary metrics in JSON
  - raw embeddings / projections in NPZ

Typical usage
-------------
uv run python src/step4_epi_visualize.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M5_r1:./contrastive_dnagpt_crop_lora_fn_s42_r1:bidir" \
  --gue-plus-dir ./data/GUE_plus \
  --epi-crop-bp 4096 \
  --epi-crop-mode junction \
  --pooling mean \
  --max-length 1024 \
  --reducer tsne \
  --output-prefix ./figures/step4_epi_tsne_m0_m5_bp4096
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np
import torch

import step4_evaluate as base


def parse_args():
    p = argparse.ArgumentParser(description="Visualise EPI embeddings from Step 4 models.")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode",
                   help="Model specs: name:path:mode")
    p.add_argument("--gue-plus-dir", required=True,
                   help="Root directory containing the EPI/ folder.")
    p.add_argument("--output-prefix", required=True,
                   help="Prefix for output files, e.g. ./figures/epi_tsne_m0_m5")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"),
                   default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epi-crop-bp", type=int, default=4096)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--epi-reverse-order", action="store_true")
    p.add_argument("--filter-n", action="store_true")
    p.add_argument(
        "--reducer",
        choices=("tsne", "umap", "pca"),
        default="tsne",
        help="2D reducer used for the plot. UMAP is optional and requires umap-learn.",
    )
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="t-SNE perplexity (used only when --reducer tsne).")
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="UMAP neighbors (used only when --reducer umap).")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist (used only when --reducer umap).")
    p.add_argument("--subsample", type=int, default=0,
                   help="Optional cap on total plotted EPI test points. 0 = use all.")
    p.add_argument(
        "--cell-line",
        default="all",
        help="Filter to one EPI cell line before visualisation. Use one of: "
             "all, epi_0, epi_1, epi_2, epi_3, epi_4, epi_5, "
             "EPI-GM12878, EPI-HeLa-S3, EPI-HUVEC, EPI-IMR90, EPI-K562, EPI-NHEK.",
    )
    p.add_argument(
        "--legend-loc",
        default="best",
        help="Matplotlib legend location for each panel.",
    )
    p.add_argument("--title", default="EPI Embedding Space")
    return p.parse_args()


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the current environment "
            "or rerun in an environment that already has matplotlib."
        ) from e


def _maybe_import_umap():
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "You requested --reducer umap, but umap-learn is not installed in the "
            "current environment. Either install it or use --reducer tsne."
        ) from e
    return umap


def _fit_2d_projection(embeddings: np.ndarray, args) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # A light PCA warm-start usually stabilises the reducers and makes t-SNE
    # noticeably less noisy on high-dimensional sequence embeddings.
    pca_dims = min(50, embeddings.shape[0], embeddings.shape[1])
    base_repr = embeddings
    if pca_dims >= 2 and embeddings.shape[1] > pca_dims:
        base_repr = PCA(n_components=pca_dims, random_state=args.seed).fit_transform(embeddings)

    if args.reducer == "pca":
        return PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)

    if args.reducer == "tsne":
        n = embeddings.shape[0]
        # sklearn requires perplexity < n_samples
        perplexity = min(args.perplexity, max(5.0, (n - 1) / 3.0))
        perplexity = min(perplexity, max(1.0, n - 1))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        return tsne.fit_transform(base_repr)

    umap = _maybe_import_umap()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    return reducer.fit_transform(base_repr)


def _centroid_distance(points: np.ndarray, labels: np.ndarray) -> float:
    uniques = np.unique(labels)
    if len(uniques) != 2:
        return float("nan")
    a = points[labels == uniques[0]].mean(axis=0)
    b = points[labels == uniques[1]].mean(axis=0)
    return float(np.linalg.norm(a - b))


def _compute_summary_metrics(embeddings: np.ndarray, proj_2d: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    metrics: dict[str, float] = {
        "n_samples": float(len(labels)),
        "positive_rate": float(labels.mean()),
        "centroid_distance_2d": _centroid_distance(proj_2d, labels),
        "centroid_distance_embedding": _centroid_distance(embeddings, labels),
    }

    if len(np.unique(labels)) == 2 and len(labels) >= 4:
        metrics["silhouette_2d"] = float(silhouette_score(proj_2d, labels))
        metrics["silhouette_embedding"] = float(silhouette_score(embeddings, labels))
        metrics["davies_bouldin_2d"] = float(davies_bouldin_score(proj_2d, labels))
    else:
        metrics["silhouette_2d"] = float("nan")
        metrics["silhouette_embedding"] = float("nan")
        metrics["davies_bouldin_2d"] = float("nan")
    return metrics


def _subsample_indices(labels: np.ndarray, cell_lines: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels))

    rng = np.random.default_rng(seed)
    keep = []
    pairs = {}
    for idx, key in enumerate(zip(labels.tolist(), cell_lines.tolist())):
        pairs.setdefault(key, []).append(idx)

    total = len(labels)
    for idxs in pairs.values():
        share = max(1, int(round(limit * len(idxs) / total)))
        share = min(share, len(idxs))
        keep.extend(rng.choice(idxs, size=share, replace=False).tolist())

    keep = np.array(sorted(set(keep)), dtype=int)
    if len(keep) > limit:
        keep = np.sort(rng.choice(keep, size=limit, replace=False))
    return keep


def _load_epi_testset(args):
    base._epi_subdir_map = dict(base._EPI_SUBDIR_DEFAULT)

    sequences: list[str] = []
    labels: list[int] = []
    cell_lines: list[str] = []
    task_keys: list[str] = []

    for task_key, _, display_name, source in base.GUE_PLUS_EPI_BENCHMARKS:
        if args.cell_line not in ("all", task_key, display_name):
            continue
        _, _, test_seqs, test_labels = base.load_benchmark_data(
            task_key,
            source,
            gue_plus_dir=args.gue_plus_dir,
            crop_bp=args.epi_crop_bp,
            crop_mode=args.epi_crop_mode,
            filter_non_acgt=args.filter_n,
            epi_reverse_order=args.epi_reverse_order,
        )
        sequences.extend(test_seqs)
        labels.extend(int(x) for x in test_labels)
        cell_lines.extend([display_name] * len(test_seqs))
        task_keys.extend([task_key] * len(test_seqs))

    labels_np = np.asarray(labels, dtype=np.int64)
    cell_lines_np = np.asarray(cell_lines)
    task_keys_np = np.asarray(task_keys)

    if len(labels_np) == 0:
        raise ValueError(
            f"No EPI rows matched --cell-line={args.cell_line!r}. "
            "Check the value or the available GUE+ EPI files."
        )

    keep = _subsample_indices(labels_np, cell_lines_np, args.subsample, args.seed)
    return {
        "sequences": [sequences[i] for i in keep],
        "labels": labels_np[keep],
        "cell_lines": cell_lines_np[keep],
        "task_keys": task_keys_np[keep],
    }


def _plot_panels(projections: dict[str, np.ndarray], labels: np.ndarray, summaries: dict[str, dict[str, float]], args):
    import matplotlib.pyplot as plt

    n_models = len(projections)
    ncols = min(2, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    colors = {0: "#4C78A8", 1: "#E45756"}
    label_names = {0: "non-interacting", 1: "interacting"}

    for ax, (model_name, proj) in zip(axes_flat, projections.items()):
        for label in sorted(np.unique(labels)):
            idx = labels == label
            ax.scatter(
                proj[idx, 0],
                proj[idx, 1],
                s=14,
                alpha=0.72,
                c=colors.get(int(label), "#777777"),
                label=label_names.get(int(label), str(label)),
                edgecolors="none",
            )
        summary = summaries[model_name]
        ax.set_title(
            f"{model_name}\n"
            f"silhouette={summary['silhouette_2d']:.3f}, "
            f"centroid-dist={summary['centroid_distance_2d']:.3f}"
        )
        ax.set_xlabel(f"{args.reducer.upper()}-1")
        ax.set_ylabel(f"{args.reducer.upper()}-2")
        ax.legend(loc=args.legend_loc, frameon=False)
        ax.grid(alpha=0.15)

    for ax in axes_flat[len(projections):]:
        ax.axis("off")

    order_text = "promoter+enhancer" if args.epi_reverse_order else "enhancer+promoter"
    fig.suptitle(
        f"{args.title}\n"
        f"EPI dev split | crop={args.epi_crop_bp}bp ({args.epi_crop_mode}) | order={order_text} | pooling={args.pooling}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    args = parse_args()
    _ensure_matplotlib()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = [base.ModelSpec.parse(s) for s in args.models]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 72)
    print("Step 4 add-on | EPI Embedding Visualisation")
    print("=" * 72)
    print(f"  Models      : {[s.name for s in specs]}")
    print(f"  Reducer     : {args.reducer}")
    print(f"  Pooling     : {args.pooling}")
    print(f"  Crop        : {args.epi_crop_bp} bp ({args.epi_crop_mode})")
    print(f"  Reverse     : {args.epi_reverse_order}")
    print(f"  Cell line   : {args.cell_line}")
    print(f"  Subsample   : {args.subsample or 'all'}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    print("\n[1/4] Loading EPI test split")
    epi = _load_epi_testset(args)
    sequences = epi["sequences"]
    labels = epi["labels"]
    cell_lines = epi["cell_lines"]
    task_keys = epi["task_keys"]
    n_cell_lines = len(np.unique(cell_lines))
    print(f"  Loaded {len(sequences):,} sequences across {n_cell_lines} cell line(s)")
    print(f"  Positive rate: {labels.mean() * 100:.2f}%")

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    projections: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    embeddings_to_save: dict[str, np.ndarray] = {}

    print("\n[2/4] Encoding and projecting")
    for spec in specs:
        print(f"\n  -> {spec.name}")
        model, tokenizer = base.load_model(spec, device, dtype)
        emb = base.encode_sequences(
            model,
            tokenizer,
            sequences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            pooling=args.pooling,
            desc=f"{spec.name} EPI",
        )
        proj = _fit_2d_projection(emb, args)
        summary = _compute_summary_metrics(emb, proj, labels)
        projections[spec.name] = proj
        summaries[spec.name] = summary
        embeddings_to_save[spec.name] = emb

        print(
            f"     silhouette_2d={summary['silhouette_2d']:.4f} | "
            f"centroid_distance_2d={summary['centroid_distance_2d']:.4f} | "
            f"davies_bouldin_2d={summary['davies_bouldin_2d']:.4f}"
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[3/4] Saving artifacts")
    np.savez_compressed(
        f"{args.output_prefix}.npz",
        labels=labels,
        cell_lines=cell_lines,
        task_keys=task_keys,
        **{f"{name}_embeddings": arr for name, arr in embeddings_to_save.items()},
        **{f"{name}_projection": arr for name, arr in projections.items()},
    )
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "models": args.models,
                    "reducer": args.reducer,
                    "pooling": args.pooling,
                    "seed": args.seed,
                    "epi_crop_bp": args.epi_crop_bp,
                    "epi_crop_mode": args.epi_crop_mode,
                    "epi_reverse_order": args.epi_reverse_order,
                    "cell_line": args.cell_line,
                    "subsample": args.subsample,
                },
                "summary": summaries,
            },
            fh,
            indent=2,
        )

    print("\n[4/4] Rendering figure")
    fig = _plot_panels(projections, labels, summaries, args)
    fig.savefig(f"{args.output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.output_prefix}.png", dpi=300, bbox_inches="tight")

    print("\nSaved:")
    print(f"  Figure : {args.output_prefix}.png / .pdf")
    print(f"  Metrics: {args.output_prefix}.json")
    print(f"  Arrays : {args.output_prefix}.npz")


if __name__ == "__main__":
    main()
