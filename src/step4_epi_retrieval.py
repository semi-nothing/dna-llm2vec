"""
Step 4 add-on: retrieval-style EPI evaluation.

This script evaluates whether an enhancer embedding retrieves its interacting
promoter from a within-task candidate pool. It is intended to test embedding
geometry directly, rather than pair-classification performance.

Example
-------
uv run python src/step4_epi_retrieval.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M5:./contrastive_dnagpt_crop_lora_fn_s42:bidir" \
  --gue-plus-dir ./data/GUE_plus \
  --task-key epi_0 \
  --split dev \
  --candidate-size 1000 \
  --pooling mean \
  --output-prefix ./eval_results/epi_retrieval_m0_m5_epi0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import step4_evaluate as base


@dataclass
class RetrievalExample:
    row_idx: int
    enhancer: str
    promoter: str
    task_key: str
    task_name: str


def parse_args():
    p = argparse.ArgumentParser(description="Run enhancer-to-promoter EPI retrieval.")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode")
    p.add_argument("--gue-plus-dir", required=True, help="Root directory containing EPI/")
    p.add_argument(
        "--task-key",
        default="all",
        choices=["all"] + [f"epi_{i}" for i in range(6)],
        help="Which EPI task to evaluate, or all six tasks.",
    )
    p.add_argument("--split", default="dev", choices=("dev", "test", "train"))
    p.add_argument("--candidate-size", type=int, default=1000,
                   help="Total candidate promoters per query, including the true promoter.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repeat negative sampling this many times and average metrics.")
    p.add_argument("--max-queries", type=int, default=None,
                   help="Optional cap on positive queries per task for quick pilots.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"),
                   default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter-n", action="store_true",
                   help="Drop rows where enhancer or promoter contains non-ACGT bases.")
    p.add_argument("--output-prefix", required=True)
    return p.parse_args()


def _task_display_name(task_key: str) -> str:
    for key, _n_classes, display_name, _source in base.GUE_PLUS_EPI_BENCHMARKS:
        if key == task_key:
            return display_name
    return task_key


def _load_epi_pairs(args, task_key: str) -> tuple[list[RetrievalExample], list[str]]:
    subdir = base._EPI_SUBDIR_DEFAULT.get(task_key, task_key)
    task_dir = os.path.join(args.gue_plus_dir, "EPI", subdir)
    path = os.path.join(task_dir, f"{args.split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing EPI split file: {path}")

    positives: list[RetrievalExample] = []
    promoter_pool: list[str] = []
    task_name = _task_display_name(task_key)
    dropped = 0

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"enhancer", "promoter", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} must contain enhancer/promoter/label columns for retrieval; "
                f"missing {sorted(missing)}"
            )

        for row_idx, row in enumerate(reader):
            enhancer = row["enhancer"].strip().upper()
            promoter = row["promoter"].strip().upper()
            if not enhancer or not promoter:
                dropped += 1
                continue
            if args.filter_n and (not base._is_acgt(enhancer) or not base._is_acgt(promoter)):
                dropped += 1
                continue

            promoter_pool.append(promoter)
            if int(row["label"]) == 1:
                positives.append(
                    RetrievalExample(
                        row_idx=row_idx,
                        enhancer=enhancer,
                        promoter=promoter,
                        task_key=task_key,
                        task_name=task_name,
                    )
                )

    if args.max_queries is not None:
        positives = positives[: args.max_queries]

    unique_promoters = sorted(set(promoter_pool))
    if dropped:
        print(f"  {task_key}: dropped {dropped} rows with empty/non-ACGT regions")
    if not positives:
        raise ValueError(f"No positive EPI pairs found for {task_key} ({args.split}).")
    if not unique_promoters:
        raise ValueError(f"No candidate promoters found for {task_key} ({args.split}).")
    return positives, unique_promoters


def _known_targets_by_enhancer(examples: list[RetrievalExample]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        known[ex.enhancer].add(ex.promoter)
    return known


def _rank_true_target(
    query: np.ndarray,
    promoter_embeddings: np.ndarray,
    true_idx: int,
    candidate_indices: np.ndarray,
) -> int:
    scores = promoter_embeddings[candidate_indices] @ query
    true_positions = np.flatnonzero(candidate_indices == true_idx)
    if true_positions.size != 1:
        raise ValueError("Candidate set must contain the true promoter exactly once.")
    true_pos = int(true_positions[0])
    true_score = float(scores[true_pos])
    # Stable tie handling: count strictly higher scores plus earlier equal scores.
    higher = int(np.sum(scores > true_score))
    equal_before = int(np.sum((scores[:true_pos] == true_score)))
    return higher + equal_before + 1


def _evaluate_retrieval_for_model(
    model_name: str,
    enhancer_embeddings: np.ndarray,
    promoter_embeddings: np.ndarray,
    examples: list[RetrievalExample],
    promoter_to_idx: dict[str, int],
    promoter_pool: list[str],
    candidate_size: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    known_targets = _known_targets_by_enhancer(examples)
    pool_indices = np.asarray([promoter_to_idx[p] for p in promoter_pool], dtype=np.int64)

    all_repeat_metrics = []
    query_records = []
    effective_candidate_sizes = []

    for repeat in range(repeats):
        ranks = []
        for q_idx, ex in enumerate(tqdm(examples, desc=f"{model_name} retrieval r{repeat + 1}", leave=False)):
            true_idx = promoter_to_idx[ex.promoter]
            forbidden = {promoter_to_idx[p] for p in known_targets[ex.enhancer]}
            negative_pool = np.asarray([idx for idx in pool_indices if idx not in forbidden], dtype=np.int64)
            n_neg = max(0, min(candidate_size - 1, len(negative_pool)))
            if n_neg:
                sampled = rng.choice(negative_pool, size=n_neg, replace=False)
                candidate_indices = np.concatenate([np.asarray([true_idx], dtype=np.int64), sampled])
            else:
                candidate_indices = np.asarray([true_idx], dtype=np.int64)
            rng.shuffle(candidate_indices)

            rank = _rank_true_target(
                enhancer_embeddings[q_idx],
                promoter_embeddings,
                true_idx,
                candidate_indices,
            )
            ranks.append(rank)
            effective_candidate_sizes.append(len(candidate_indices))
            if repeat == 0:
                query_records.append(
                    {
                        "task_key": ex.task_key,
                        "row_idx": ex.row_idx,
                        "rank": rank,
                        "candidate_size": len(candidate_indices),
                    }
                )

        ranks_arr = np.asarray(ranks, dtype=np.float64)
        metrics = {
            "top1": float(np.mean(ranks_arr <= 1)),
            "top5": float(np.mean(ranks_arr <= 5)),
            "top10": float(np.mean(ranks_arr <= 10)),
            "mrr": float(np.mean(1.0 / ranks_arr)),
            "median_rank": float(np.median(ranks_arr)),
            "mean_rank": float(np.mean(ranks_arr)),
        }
        all_repeat_metrics.append(metrics)

    summary = {
        key: float(np.mean([m[key] for m in all_repeat_metrics]))
        for key in all_repeat_metrics[0]
    }
    summary.update(
        {
            f"{key}_std": float(np.std([m[key] for m in all_repeat_metrics], ddof=0))
            for key in all_repeat_metrics[0]
        }
    )
    summary["n_queries"] = len(examples)
    summary["candidate_size_requested"] = candidate_size
    summary["candidate_size_effective_mean"] = float(np.mean(effective_candidate_sizes))
    summary["repeats"] = repeats
    return {
        "summary": summary,
        "repeat_metrics": all_repeat_metrics,
        "queries_first_repeat": query_records,
    }


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "model",
        "task_key",
        "task_name",
        "n_queries",
        "candidate_size_effective_mean",
        "top1",
        "top5",
        "top10",
        "mrr",
        "median_rank",
        "mean_rank",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = [base.ModelSpec.parse(s) for s in args.models]
    task_keys = [f"epi_{i}" for i in range(6)] if args.task_key == "all" else [args.task_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 72)
    print("Step 4 add-on | EPI enhancer-to-promoter retrieval")
    print("=" * 72)
    print(f"  Models      : {[s.name for s in specs]}")
    print(f"  Tasks       : {task_keys}")
    print(f"  Split       : {args.split}")
    print(f"  Candidates  : {args.candidate_size} (including true promoter)")
    print(f"  Repeats     : {args.repeats}")
    print(f"  Pooling     : {args.pooling}")
    print(f"  Device      : {device}")
    print("=" * 72)

    task_payloads = {}
    for task_key in task_keys:
        examples, promoter_pool = _load_epi_pairs(args, task_key)
        task_payloads[task_key] = (examples, promoter_pool)
        print(
            f"  {task_key:<5} {examples[0].task_name:<12} "
            f"queries={len(examples):>5} promoters={len(promoter_pool):>5}"
        )

    results: dict[str, Any] = {
        "config": vars(args),
        "tasks": {},
        "models": [spec.__dict__ for spec in specs],
    }
    summary_rows: list[dict[str, Any]] = []

    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer = base.load_model(spec, device, dtype)

        for task_key in task_keys:
            examples, promoter_pool = task_payloads[task_key]
            unique_enhancers = [ex.enhancer for ex in examples]
            unique_promoters = promoter_pool
            promoter_to_idx = {promoter: i for i, promoter in enumerate(unique_promoters)}

            enhancer_embeddings = base.encode_sequences(
                model,
                tokenizer,
                unique_enhancers,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
                pooling=args.pooling,
                desc=f"{spec.name} {task_key} enhancers",
            )
            promoter_embeddings = base.encode_sequences(
                model,
                tokenizer,
                unique_promoters,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
                pooling=args.pooling,
                desc=f"{spec.name} {task_key} promoters",
            )

            model_result = _evaluate_retrieval_for_model(
                model_name=spec.name,
                enhancer_embeddings=enhancer_embeddings,
                promoter_embeddings=promoter_embeddings,
                examples=examples,
                promoter_to_idx=promoter_to_idx,
                promoter_pool=promoter_pool,
                candidate_size=args.candidate_size,
                repeats=args.repeats,
                seed=args.seed,
            )
            summary = model_result["summary"]
            print(
                f"   {task_key:<5} Top1={summary['top1']*100:6.2f} "
                f"Top10={summary['top10']*100:6.2f} MRR={summary['mrr']:.4f} "
                f"median_rank={summary['median_rank']:.1f}"
            )

            results["tasks"].setdefault(task_key, {})[spec.name] = model_result
            summary_rows.append(
                {
                    "model": spec.name,
                    "task_key": task_key,
                    "task_name": examples[0].task_name,
                    **summary,
                }
            )

            del enhancer_embeddings, promoter_embeddings
            if device == "cuda":
                torch.cuda.empty_cache()

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    overall: dict[str, Any] = {}
    for spec in specs:
        model_rows = [row for row in summary_rows if row["model"] == spec.name]
        if len(model_rows) <= 1:
            continue
        weights = np.asarray([row["n_queries"] for row in model_rows], dtype=np.float64)
        total = float(weights.sum())
        aggregate = {
            "model": spec.name,
            "task_key": "all",
            "task_name": "EPI-Avg",
            "n_queries": int(total),
            "candidate_size_effective_mean": float(
                np.average([row["candidate_size_effective_mean"] for row in model_rows], weights=weights)
            ),
        }
        for key in ("top1", "top5", "top10", "mrr", "median_rank", "mean_rank"):
            aggregate[key] = float(np.average([row[key] for row in model_rows], weights=weights))
        summary_rows.append(aggregate)
        overall[spec.name] = aggregate
    results["overall"] = overall

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _write_summary_csv(f"{args.output_prefix}.csv", summary_rows)

    print("\nSaved:")
    print(f"  JSON summary: {args.output_prefix}.json")
    print(f"  CSV summary : {args.output_prefix}.csv")


if __name__ == "__main__":
    main()
