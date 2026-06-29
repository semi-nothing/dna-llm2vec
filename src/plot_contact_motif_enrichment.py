"""
Plot contact-motif enrichment summaries from analyze_contact_motif_enrichment.py.

Example:
  uv run python src/plot_contact_motif_enrichment.py \
    --prefix figures/bp_jacobian_res/hg38_motif_strict_len6_flank0 \
    --output-dir figures/bp_jacobian_res/plots
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "m0": "M0\nbase",
    "m1": "M1\n+bidir",
    "m2": "M2\n+MNTP",
    "m3": "M3\ndropout",
    "m4": "M4\nrevcomp",
    "m5": "M5\ncrop",
    "m6": "M6\nshift",
    "h0": "H0\nbase",
    "h1": "H1\n+bidir",
    "h2": "H2\n+MNTP",
    "h3": "H3\ndropout",
    "h4": "H4\nrevcomp",
    "h5": "H5\ncrop",
    "h6": "H6\nshift",
}

MODEL_COLORS = {
    "m0": "#59636f",
    "m1": "#3b82f6",
    "m2": "#0f766e",
    "m3": "#7c3aed",
    "m4": "#dc2626",
    "m5": "#ea580c",
    "m6": "#64748b",
    "h0": "#59636f",
    "h1": "#3b82f6",
    "h2": "#0f766e",
    "h3": "#7c3aed",
    "h4": "#dc2626",
    "h5": "#ea580c",
    "h6": "#64748b",
}


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def model_sort_key(model: str):
    prefix = "".join(ch for ch in model if not ch.isdigit())
    digits = "".join(ch for ch in model if ch.isdigit())
    return prefix, int(digits) if digits else 999


def model_names(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["model_variant"] for row in rows}, key=model_sort_key)


def setup_axes(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall(aggregate_rows: list[dict[str, str]], output_dir: str):
    rows_by_model = {row["model_variant"]: row for row in aggregate_rows}
    models = model_names(aggregate_rows)
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(model.lower(), "#4b5563") for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    enrich = [f(rows_by_model[m], "mean_motif_pair_enrichment") for m in models]
    same = [f(rows_by_model[m], "mean_same_motif_pair_enrichment") for m in models]
    axes[0].bar(x - 0.18, enrich, width=0.36, color=colors, alpha=0.85, label="Motif pair")
    axes[0].bar(x + 0.18, same, width=0.36, color=colors, alpha=0.45, label="Same motif")
    axes[0].axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [MODEL_LABELS.get(m.lower(), m.upper()) for m in models])
    setup_axes(axes[0], "Strict motif enrichment", "Mean enrichment")
    axes[0].legend(frameon=False, fontsize=9)

    z = [f(rows_by_model[m], "stouffer_z") for m in models]
    same_z = [f(rows_by_model[m], "mean_same_motif_pair_z") for m in models]
    axes[1].plot(x, z, marker="o", linewidth=2.2, color="#111827", label="Motif pair Stouffer z")
    axes[1].plot(x, same_z, marker="s", linewidth=2.0, color="#dc2626", label="Mean same-motif z")
    if "m4" in [m.lower() for m in models] or "h4" in [m.lower() for m in models]:
        peak_idx = max(range(len(models)), key=lambda idx: z[idx])
        axes[1].annotate("peak", (x[peak_idx], z[peak_idx]), xytext=(0, 10),
                         textcoords="offset points", ha="center", color="#dc2626")
    axes[1].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [MODEL_LABELS.get(m.lower(), m.upper()) for m in models])
    setup_axes(axes[1], "Pooled significance across windows", "z")
    axes[1].legend(frameon=False, fontsize=9)

    savefig(fig, output_dir, "motif_enrichment_overall")


def plot_groups(group_rows: list[dict[str, str]], output_dir: str):
    groups = sorted({row["group"] for row in group_rows})
    models = sorted({row["model_variant"] for row in group_rows}, key=model_sort_key)
    row_by_key = {(row["model_variant"], row["group"]): row for row in group_rows}
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    palette = {"promoter": "#2563eb", "tfbs": "#dc2626", "insulator": "#7c3aed"}

    for group in groups:
        y = [f(row_by_key[(model, group)], "group_stouffer_z") for model in models]
        axes[0].plot(x, y, marker="o", linewidth=2.0, label=group, color=palette.get(group, None))
    axes[0].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [MODEL_LABELS.get(m.lower(), m.upper()) for m in models])
    setup_axes(axes[0], "Motif group signal", "Stouffer z")
    axes[0].legend(frameon=False, fontsize=9)

    for group in groups:
        y = [f(row_by_key[(model, group)], "group_same_motif_stouffer_z") for model in models]
        axes[1].plot(x, y, marker="s", linewidth=2.0, label=group, color=palette.get(group, None))
    axes[1].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [MODEL_LABELS.get(m.lower(), m.upper()) for m in models])
    setup_axes(axes[1], "Same-motif group signal", "Stouffer z")
    axes[1].legend(frameon=False, fontsize=9)

    savefig(fig, output_dir, "motif_enrichment_groups")


def window_id(row: dict[str, str]) -> str:
    chrom = row.get("chrom", "")
    start = row.get("window_start_1based", "")
    end = row.get("window_end_1based", "")
    if chrom and start and end:
        return f"{chrom}:{start}-{end}"
    label = row["label"]
    parts = label.split("_", 1)
    return parts[1] if len(parts) > 1 else label


def plot_paired(summary_rows: list[dict[str, str]], output_dir: str, ref_model: str, test_model: str):
    by_window: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in summary_rows:
        by_window[window_id(row)][row["model_variant"].lower()] = row

    pairs = []
    for win, items in by_window.items():
        if ref_model in items and test_model in items:
            pairs.append((win, items[ref_model], items[test_model]))
    if not pairs:
        return

    ref = np.array([f(row_ref, "motif_pair_enrichment") for _, row_ref, _ in pairs])
    test = np.array([f(row_test, "motif_pair_enrichment") for _, _, row_test in pairs])
    same_ref = np.array([f(row_ref, "same_motif_pair_enrichment") for _, row_ref, _ in pairs])
    same_test = np.array([f(row_test, "same_motif_pair_enrichment") for _, _, row_test in pairs])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    lim_min = float(np.nanmin([ref.min(), test.min(), 1.0])) - 0.03
    lim_max = float(np.nanmax([ref.max(), test.max(), 1.0])) + 0.03
    axes[0].scatter(ref, test, color=MODEL_COLORS.get(test_model, "#dc2626"), alpha=0.8)
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(lim_min, lim_max)
    axes[0].set_ylim(lim_min, lim_max)
    axes[0].set_xlabel(f"{ref_model.upper()} enrichment")
    axes[0].set_ylabel(f"{test_model.upper()} enrichment")
    setup_axes(axes[0], "Per-window motif pair enrichment", " ")

    diff = test - ref
    same_diff = same_test - same_ref
    axes[1].boxplot([diff, same_diff], labels=["Motif pair", "Same motif"], widths=0.5, patch_artist=True,
                    boxprops={"facecolor": "#fee2e2", "color": "#991b1b"},
                    medianprops={"color": "#111827"})
    axes[1].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    setup_axes(axes[1], f"{test_model.upper()} - {ref_model.upper()} paired delta", "Enrichment delta")

    savefig(fig, output_dir, f"motif_enrichment_paired_{test_model}_vs_{ref_model}")


def plot_ablation_story(aggregate_rows: list[dict[str, str]], group_rows: list[dict[str, str]], output_dir: str):
    rows_by_model = {row["model_variant"].lower(): row for row in aggregate_rows}
    model_prefix = None
    for prefix in ("m", "h"):
        required = {f"{prefix}{idx}" for idx in range(7)}
        if required.issubset(rows_by_model):
            model_prefix = prefix
            break
    if model_prefix is None:
        return

    fig = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axes = [
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[0, 1]),
        fig.add_subplot(spec[1, :]),
    ]

    chain = [f"{model_prefix}{idx}" for idx in range(3)]
    chain_z = [f(rows_by_model[m], "stouffer_z") for m in chain]
    x = np.arange(len(chain))
    axes[0].plot(x, chain_z, marker="o", linewidth=2.5, color="#111827")
    axes[0].fill_between(x, chain_z, 0, color="#dbeafe", alpha=0.8)
    for idx, (model, z) in enumerate(zip(chain, chain_z)):
        axes[0].text(idx, z + max(chain_z) * 0.04, f"{z:.1f}", ha="center", fontsize=9)
    for idx in range(1, len(chain)):
        delta = chain_z[idx] - chain_z[idx - 1]
        y_mid = (chain_z[idx] + chain_z[idx - 1]) / 2
        axes[0].annotate(
            f"{delta:+.1f}",
            xy=(idx - 0.5, y_mid),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#2563eb",
            fontsize=9,
        )
    axes[0].set_xticks(x, [MODEL_LABELS[m] for m in chain])
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].set_ylim(0, max(chain_z) * 1.22)
    setup_axes(axes[0], "A. Architecture adaptation", "Motif-pair Stouffer z")

    variants = [f"{model_prefix}{idx}" for idx in range(3, 7)]
    baseline_model = f"{model_prefix}2"
    baseline = f(rows_by_model[baseline_model], "stouffer_z")
    deltas = [f(rows_by_model[m], "stouffer_z") - baseline for m in variants]
    x2 = np.arange(len(variants))
    colors = [MODEL_COLORS[m] for m in variants]
    axes[1].bar(x2, deltas, color=colors, alpha=0.9)
    axes[1].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    delta_min = min(deltas + [0.0])
    delta_max = max(deltas + [0.0])
    delta_pad = max(1.4, 0.18 * (delta_max - delta_min))
    axes[1].set_ylim(delta_min - delta_pad, delta_max + delta_pad)
    label_step = 0.04 * (delta_max - delta_min)
    for idx, delta in enumerate(deltas):
        va = "bottom" if delta >= 0 else "top"
        offset = label_step if delta >= 0 else -label_step
        axes[1].text(idx, delta + offset, f"{delta:+.1f}", ha="center", va=va, fontsize=9)
    axes[1].set_xticks(x2, [MODEL_LABELS[m] for m in variants])
    axes[1].tick_params(axis="x", labelsize=9)
    setup_axes(axes[1], f"B. Contrastive variants vs {baseline_model.upper()}", "Delta Stouffer z")

    group_by_key = {(row["model_variant"].lower(), row["group"]): row for row in group_rows}
    group_models = [f"{model_prefix}{idx}" for idx in range(2, 7)]
    x3 = np.arange(len(group_models))
    group_values = []
    for group, color in (("promoter", "#2563eb"), ("tfbs", "#dc2626")):
        if not all((model, group) in group_by_key for model in group_models):
            continue
        y = [f(group_by_key[(model, group)], "group_stouffer_z") for model in group_models]
        group_values.extend(y)
        axes[2].plot(x3, y, marker="o", linewidth=2.3, label=group.upper(), color=color)
        peak = int(np.nanargmax(y))
        label_offset = 0.04 * (max(y) - min(y))
        axes[2].text(peak, y[peak] + label_offset, f"{y[peak]:.1f}", ha="center", fontsize=9, color=color)
    if group_values:
        group_min = min(group_values + [0.0])
        group_max = max(group_values + [0.0])
        group_pad = max(1.8, 0.18 * (group_max - group_min))
        axes[2].set_ylim(group_min - group_pad, group_max + group_pad)
    axes[2].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(x3, [MODEL_LABELS[m] for m in group_models])
    axes[2].tick_params(axis="x", labelsize=9)
    setup_axes(axes[2], "C. Group-specific contrastive signal", "Group Stouffer z")
    axes[2].legend(frameon=False, fontsize=9)

    savefig(fig, output_dir, "motif_enrichment_ablation_story")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True, help="Output prefix used by analyze_contact_motif_enrichment.py")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ref-model", default="m0")
    p.add_argument("--test-model", default="m4")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.prefix))
    aggregate = read_csv(f"{args.prefix}_aggregate_model_variant.csv")
    group = read_csv(f"{args.prefix}_aggregate_groups_model_variant.csv")
    summary = read_csv(f"{args.prefix}_summary.csv")
    plot_overall(aggregate, output_dir)
    plot_groups(group, output_dir)
    plot_paired(summary, output_dir, args.ref_model.lower(), args.test_model.lower())
    plot_ablation_story(aggregate, group, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
