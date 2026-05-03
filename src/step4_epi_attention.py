"""
Step 4 add-on: EPI attention case-study visualisation.

This script creates a compact mechanistic figure for EPI examples by plotting
last-layer attention maps for selected models (typically M0 vs M5).

Default workflow:
  - load one GUE+ EPI task (cell line) from the dev split
  - pick one interacting example and one non-interacting example
  - extract last-layer attentions
  - average over heads
  - downsample the attention map for readability
  - save a 2 x N panel heatmap, plus JSON/NPZ summaries

Example
-------
uv run python src/step4_epi_attention.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M5_r1:./contrastive_dnagpt_crop_lora_fn_s42_r1:bidir" \
  --gue-plus-dir ./data/GUE_plus \
  --task-key epi_0 \
  --epi-crop-bp 4096 \
  --epi-crop-mode junction \
  --filter-n \
  --max-length 1024 \
  --output-prefix ./figures/step4_epi_attention_m0_m5_epi0_bp4096
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import torch

import step4_evaluate as base


def parse_args():
    p = argparse.ArgumentParser(description="Visualise last-layer EPI attention maps.")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode",
                   help="Model specs: name:path:mode")
    p.add_argument("--gue-plus-dir", required=True,
                   help="Root directory containing the EPI/ folder.")
    p.add_argument("--task-key", default="epi_0",
                   choices=[f"epi_{i}" for i in range(6)],
                   help="Which EPI task / cell line to visualise.")
    p.add_argument("--split", default="dev", choices=("dev", "train", "test"))
    p.add_argument("--positive-index", type=int, default=0,
                   help="Which positive example to use within the chosen split.")
    p.add_argument("--negative-index", type=int, default=0,
                   help="Which negative example to use within the chosen split.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Kept for interface consistency; attention plots use single-example forward passes.")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epi-crop-bp", type=int, default=4096)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--epi-reverse-order", action="store_true")
    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--downsample-bins", type=int, default=96,
                   help="Downsample attention maps to this many bins per axis for plotting.")
    p.add_argument("--head-mode", choices=("mean", "head0"), default="mean",
                   help="How to reduce heads. mean is most stable for paper figures.")
    p.add_argument("--color-scale",
                   choices=("global", "robust_global", "panel", "robust_panel"),
                   default="robust_global",
                   help="How to choose the heatmap colour scale. robust_* uses a percentile cap "
                        "to avoid one sharp hotspot making other panels look black.")
    p.add_argument("--color-percentile", type=float, default=99.5,
                   help="Percentile used by robust_* colour scaling modes.")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--title", default="EPI Attention Case Study")
    return p.parse_args()


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the current environment "
            "or rerun in an environment that already has matplotlib."
        ) from e


def _load_epi_rows(args) -> list[dict[str, Any]]:
    base._epi_subdir_map = dict(base._EPI_SUBDIR_DEFAULT)
    subdir = base._epi_subdir_map.get(args.task_key, base._EPI_SUBDIR_DEFAULT[args.task_key])
    task_dir = os.path.join(args.gue_plus_dir, "EPI", subdir)
    path = os.path.join(task_dir, f"{args.split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing EPI split file: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            seq = (row.get("sequence") or row.get("seq") or "").strip().upper()
            enhancer = row.get("enhancer")
            promoter = row.get("promoter")

            anchor = None
            if not seq and enhancer is not None and promoter is not None:
                enhancer = enhancer.strip().upper()
                promoter = promoter.strip().upper()
                if args.epi_reverse_order:
                    seq = promoter + enhancer
                    anchor = len(promoter)
                else:
                    seq = enhancer + promoter
                    anchor = len(enhancer)

            if not seq:
                continue

            cropped = base._crop_sequence(seq, args.epi_crop_bp, args.epi_crop_mode, anchor)
            if anchor is not None and args.epi_crop_mode == "junction":
                start = anchor - (args.epi_crop_bp // 2)
                start = max(0, min(start, len(seq) - args.epi_crop_bp))
                cropped_anchor = anchor - start
                cropped_anchor = max(0, min(cropped_anchor, len(cropped)))
            elif anchor is not None and args.epi_crop_mode == "center":
                start = (len(seq) - min(len(seq), args.epi_crop_bp)) // 2
                cropped_anchor = anchor - start
                cropped_anchor = max(0, min(cropped_anchor, len(cropped)))
            else:
                cropped_anchor = len(cropped) // 2

            if args.filter_n and not base._is_acgt(cropped):
                continue

            rows.append(
                {
                    "row_idx": row_idx,
                    "sequence": cropped,
                    "label": int(row["label"]),
                    "anchor_bp": int(cropped_anchor),
                    "subdir": subdir,
                }
            )
    return rows


def _select_examples(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    if not pos:
        raise ValueError("No positive examples found after filtering/cropping.")
    if not neg:
        raise ValueError("No negative examples found after filtering/cropping.")
    if args.positive_index >= len(pos):
        raise IndexError(f"--positive-index={args.positive_index} but only {len(pos)} positives are available.")
    if args.negative_index >= len(neg):
        raise IndexError(f"--negative-index={args.negative_index} but only {len(neg)} negatives are available.")
    return [pos[args.positive_index], neg[args.negative_index]]


def _tokenize_one(tokenizer, sequence: str, max_length: int, device: str):
    enc = tokenizer(
        sequence,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _boundary_token_index(tokenizer, sequence: str, anchor_bp: int, max_length: int) -> int:
    left_seq = sequence[:anchor_bp]
    left_ids = tokenizer(
        left_seq,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]
    return int(min(len(left_ids), max_length))


@torch.no_grad()
def _extract_last_attention(model, tokenizer, sequence: str, max_length: int, device: str, head_mode: str):
    enc = _tokenize_one(tokenizer, sequence, max_length, device)
    if hasattr(model, "transformer"):
        out = model.transformer(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    else:
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if not hasattr(out, "attentions") or out.attentions is None:
        raise RuntimeError("Model output did not include attentions.")

    attn = out.attentions[-1]  # (B, H, T, T)
    if isinstance(attn, tuple):
        attn = attn[0]
    attn = attn[0].detach().float().cpu().numpy()  # (H, T, T)

    if head_mode == "mean":
        attn_2d = attn.mean(axis=0)
    else:
        attn_2d = attn[0]

    token_count = int(enc["attention_mask"][0].sum().item())
    attn_2d = attn_2d[:token_count, :token_count]
    input_ids = enc["input_ids"][0, :token_count].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return attn_2d, tokens, input_ids


def _downsample_attention(attn: np.ndarray, bins: int) -> np.ndarray:
    n = attn.shape[0]
    if bins <= 0 or bins >= n:
        return attn

    edges = np.linspace(0, n, bins + 1, dtype=int)
    edges[0] = 0
    edges[-1] = n
    pooled = np.zeros((bins, bins), dtype=np.float32)
    for i in range(bins):
        rs, re = edges[i], edges[i + 1]
        for j in range(bins):
            cs, ce = edges[j], edges[j + 1]
            block = attn[rs:re, cs:ce]
            pooled[i, j] = float(block.mean()) if block.size else 0.0
    return pooled


def _cross_junction_metrics(attn: np.ndarray, boundary: int) -> dict[str, float]:
    n = attn.shape[0]
    boundary = max(1, min(boundary, n - 1))
    left = slice(0, boundary)
    right = slice(boundary, n)

    left_to_right = float(attn[left, right].mean()) if boundary < n else float("nan")
    right_to_left = float(attn[right, left].mean()) if boundary > 0 else float("nan")
    within_left = float(attn[left, left].mean())
    within_right = float(attn[right, right].mean()) if boundary < n else float("nan")
    cross_mean = float(np.nanmean([left_to_right, right_to_left]))
    within_mean = float(np.nanmean([within_left, within_right]))
    ratio = float(cross_mean / within_mean) if within_mean and not np.isnan(within_mean) else float("nan")

    return {
        "left_to_right_mean": left_to_right,
        "right_to_left_mean": right_to_left,
        "within_left_mean": within_left,
        "within_right_mean": within_right,
        "cross_mean": cross_mean,
        "within_mean": within_mean,
        "cross_over_within_ratio": ratio,
    }


def _plot_attention_grid(results: dict[str, dict[str, Any]], examples: list[dict[str, Any]], args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    model_names = list(results.keys())
    nrows = len(examples)
    ncols = len(model_names)
    fig = plt.figure(figsize=(5.5 * ncols + 3.8, 5 * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.82],
        wspace=0.36,
        hspace=0.42,
    )
    axes = np.asarray(
        [[fig.add_subplot(gs[row, col]) for col in range(ncols)] for row in range(nrows)],
        dtype=object,
    )
    quant_ax = fig.add_subplot(gs[:, -1])

    heatmaps = [
        np.asarray(model_result[ex_key]["heatmap"], dtype=np.float32)
        for model_result in results.values()
        for ex_key in model_result
    ]

    def _panel_vmax(heatmap: np.ndarray) -> float:
        if args.color_scale == "panel":
            return max(float(heatmap.max()), 1e-6)
        if args.color_scale == "robust_panel":
            return max(float(np.percentile(heatmap, args.color_percentile)), 1e-6)
        if args.color_scale == "robust_global":
            return max(float(np.percentile(np.concatenate([h.ravel() for h in heatmaps]), args.color_percentile)), 1e-6)
        return max(float(max(h.max() for h in heatmaps)), 1e-6)

    shared_vmax = None
    if args.color_scale in {"global", "robust_global"}:
        shared_vmax = _panel_vmax(heatmaps[0])

    im = None
    for col, model_name in enumerate(model_names):
        for row, example in enumerate(examples):
            ex_key = f"label_{example['label']}"
            item = results[model_name][ex_key]
            heatmap = item["heatmap"]
            boundary_bin = item["boundary_bin"]
            ax = axes[row, col]
            vmax = shared_vmax if shared_vmax is not None else _panel_vmax(heatmap)
            im = ax.imshow(heatmap, cmap="magma", vmin=0.0, vmax=vmax, origin="lower")
            n_bins = heatmap.shape[0]
            ax.axvline(boundary_bin - 0.5, color="cyan", linewidth=1.4, linestyle="--")
            ax.axhline(boundary_bin - 0.5, color="cyan", linewidth=1.4, linestyle="--")

            # Highlight the two off-diagonal enhancer-promoter cross-junction quadrants.
            cross_color = "#00e5ff"
            ax.add_patch(
                Rectangle(
                    (boundary_bin - 0.5, -0.5),
                    n_bins - boundary_bin,
                    boundary_bin,
                    fill=False,
                    edgecolor=cross_color,
                    linewidth=2.0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (-0.5, boundary_bin - 0.5),
                    boundary_bin,
                    n_bins - boundary_bin,
                    fill=False,
                    edgecolor=cross_color,
                    linewidth=2.0,
                )
            )
            ax.text(
                boundary_bin + (n_bins - boundary_bin) / 2,
                boundary_bin / 2,
                "E->P",
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
            )
            ax.text(
                boundary_bin / 2,
                boundary_bin + (n_bins - boundary_bin) / 2,
                "P->E",
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
            )

            enhancer_center = max((boundary_bin - 1) / 2, 0)
            promoter_center = boundary_bin + max((n_bins - boundary_bin - 1) / 2, 0)
            ax.set_xticks([enhancer_center, promoter_center])
            ax.set_xticklabels(["Enhancer", "Promoter"], rotation=0)
            ax.set_yticks([enhancer_center, promoter_center])
            ax.set_yticklabels(["Enhancer", "Promoter"], rotation=90, va="center")
            label_name = "interacting" if example["label"] == 1 else "non-interacting"
            ratio = item["metrics"]["cross_over_within_ratio"]
            ax.set_title(
                f"{model_name} | {label_name}\n"
                f"cross/within={ratio:.3f}"
            )
            ax.set_xlabel("Key region")
            ax.set_ylabel("Query region")

    label_names = [
        "interacting" if example["label"] == 1 else "non-interacting"
        for example in examples
    ]
    colors = {
        "interacting": "#e15759",
        "non-interacting": "#4e79a7",
    }
    x = np.arange(len(model_names), dtype=float)
    width = min(0.34, 0.7 / max(len(examples), 1))
    offsets = (np.arange(len(examples)) - (len(examples) - 1) / 2) * width
    for ex_idx, example in enumerate(examples):
        ex_key = f"label_{example['label']}"
        label_name = label_names[ex_idx]
        ratios = [
            results[model_name][ex_key]["metrics"]["cross_over_within_ratio"]
            for model_name in model_names
        ]
        quant_ax.bar(
            x + offsets[ex_idx],
            ratios,
            width=width,
            color=colors.get(label_name, "0.5"),
            alpha=0.85,
            label=label_name,
        )
        quant_ax.scatter(
            x + offsets[ex_idx],
            ratios,
            s=22,
            color="black",
            zorder=3,
            linewidth=0,
        )
    quant_ax.axhline(1.0, color="0.35", linewidth=1.0, linestyle="--")
    quant_ax.set_title("Cross-junction\nattention ratio")
    quant_ax.set_ylabel("Cross / within")
    quant_ax.set_xticks(x)
    quant_ax.set_xticklabels(model_names, rotation=45, ha="right")
    quant_ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    quant_ax.legend(frameon=False, fontsize=8)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        if args.color_scale in {"panel", "robust_panel"}:
            cbar.set_label("Last-layer attention (panel-scaled)")
        else:
            cbar.set_label("Last-layer attention")

    order_text = "promoter+enhancer" if args.epi_reverse_order else "enhancer+promoter"
    fig.suptitle(
        f"{args.title}\n"
        f"{args.task_key} | {args.split} split | crop={args.epi_crop_bp}bp ({args.epi_crop_mode}) | "
        f"order={order_text} | scale={args.color_scale}",
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
    print("Step 4 add-on | EPI Attention Case Study")
    print("=" * 72)
    print(f"  Models      : {[s.name for s in specs]}")
    print(f"  Task        : {args.task_key} ({args.split})")
    print(f"  Crop        : {args.epi_crop_bp} bp ({args.epi_crop_mode})")
    print(f"  Reverse     : {args.epi_reverse_order}")
    print(f"  Head mode   : {args.head_mode}")
    print(f"  Downsample  : {args.downsample_bins}")
    print(f"  Color scale : {args.color_scale} (p={args.color_percentile:g})")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    rows = _load_epi_rows(args)
    examples = _select_examples(rows, args)

    print("\nSelected examples:")
    for ex in examples:
        label_name = "interacting" if ex["label"] == 1 else "non-interacting"
        print(
            f"  {label_name:<16} row={ex['row_idx']} | "
            f"len={len(ex['sequence'])} bp | anchor={ex['anchor_bp']} bp"
        )

    results: dict[str, dict[str, Any]] = {}
    arrays_to_save: dict[str, np.ndarray] = {}

    for spec in specs:
        print(f"\n-> {spec.name}")
        if spec.mode == "encoder":
            raise SystemExit("This script currently targets GPT-style causal/bidir models, not encoder specs.")
        model, tokenizer = base.load_model(spec, device, dtype)
        model.eval()
        results[spec.name] = {}

        for ex in examples:
            ex_key = f"label_{ex['label']}"
            attn, tokens, input_ids = _extract_last_attention(
                model,
                tokenizer,
                ex["sequence"],
                args.max_length,
                device,
                args.head_mode,
            )
            boundary_token = _boundary_token_index(tokenizer, ex["sequence"], ex["anchor_bp"], args.max_length)
            boundary_token = max(1, min(boundary_token, attn.shape[0] - 1))
            heatmap = _downsample_attention(attn, args.downsample_bins)
            boundary_bin = int(round(boundary_token * heatmap.shape[0] / attn.shape[0]))
            boundary_bin = max(1, min(boundary_bin, heatmap.shape[0] - 1))
            metrics = _cross_junction_metrics(attn, boundary_token)

            print(
                f"   label={ex['label']} | tokens={attn.shape[0]} | "
                f"cross/within={metrics['cross_over_within_ratio']:.4f}"
            )

            results[spec.name][ex_key] = {
                "heatmap": heatmap,
                "boundary_bin": boundary_bin,
                "boundary_token": boundary_token,
                "token_count": attn.shape[0],
                "metrics": metrics,
                "row_idx": ex["row_idx"],
                "tokens_preview": tokens[:32],
                "input_ids_preview": input_ids[:32],
            }
            arrays_to_save[f"{spec.name}_{ex_key}_attention"] = attn
            arrays_to_save[f"{spec.name}_{ex_key}_heatmap"] = heatmap

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    fig = _plot_attention_grid(results, examples, args)
    fig.savefig(f"{args.output_prefix}.png", dpi=300, bbox_inches="tight")

    json_ready = {
        "config": {
            "models": args.models,
            "task_key": args.task_key,
            "split": args.split,
            "epi_crop_bp": args.epi_crop_bp,
            "epi_crop_mode": args.epi_crop_mode,
            "epi_reverse_order": args.epi_reverse_order,
            "head_mode": args.head_mode,
            "downsample_bins": args.downsample_bins,
            "color_scale": args.color_scale,
            "color_percentile": args.color_percentile,
            "positive_index": args.positive_index,
            "negative_index": args.negative_index,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "examples": [
            {
                "label": ex["label"],
                "row_idx": ex["row_idx"],
                "anchor_bp": ex["anchor_bp"],
                "sequence_length_bp": len(ex["sequence"]),
                "subdir": ex["subdir"],
            }
            for ex in examples
        ],
        "results": {
            model_name: {
                ex_key: {
                    k: v
                    for k, v in item.items()
                    if k not in ("heatmap", "tokens_preview", "input_ids_preview")
                }
                for ex_key, item in model_result.items()
            }
            for model_name, model_result in results.items()
        },
    }
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump(json_ready, fh, indent=2)

    np.savez_compressed(
        f"{args.output_prefix}.npz",
        **arrays_to_save,
        labels=np.asarray([ex["label"] for ex in examples], dtype=np.int64),
        row_indices=np.asarray([ex["row_idx"] for ex in examples], dtype=np.int64),
        anchor_bp=np.asarray([ex["anchor_bp"] for ex in examples], dtype=np.int64),
    )

    print("\nSaved:")
    print(f"  Figure : {args.output_prefix}.png")
    print(f"  Metrics: {args.output_prefix}.json")
    print(f"  Arrays : {args.output_prefix}.npz")


if __name__ == "__main__":
    main()
