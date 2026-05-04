"""
Step 4 add-on: retrieval-style EPI evaluation.

This script evaluates whether an enhancer embedding retrieves its interacting
promoter from a within-task candidate pool. It is intended to test embedding
geometry directly, rather than pair-classification performance.

Two candidate-pool modes are supported (--candidate-pool):
  - ``global`` (default): the candidate set per query is sampled from the
    union of every promoter that appears anywhere in the split. This is the
    ``broad'' retrieval setting; it is easier because most candidates are
    far from the query.
  - ``per-enhancer``: the candidate set is the set of promoters that appear
    on the same enhancer's rows in the data file (typically the harder
    enhancer-specific negatives baked into the EPI dataset).

Replace the example checkpoint paths below with paths that exist on your
machine before running.

Example
-------
uv run python src/step4_epi_retrieval.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M5:/abs/path/to/your/M5_checkpoint:bidir" \
  --gue-plus-dir ./data/GUE_plus \
  --task-key epi_0 \
  --split dev \
  --candidate-size 1000 \
  --candidate-pool global \
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
                   help="Total candidate promoters per query, including the true promoter. "
                        "If the available candidate pool is smaller than this, the effective "
                        "candidate size shrinks; check candidate_size_effective_mean in the output.")
    p.add_argument("--candidate-pool", choices=("global", "per-enhancer"), default="global",
                   help="global: sample negatives from all promoters in the split (broad, easier). "
                        "per-enhancer: sample only from the promoters paired with this enhancer in "
                        "the data file (matches the original EPI candidate set, harder).")
    p.add_argument("--allow-short-candidates", action="store_true",
                   help="Allow queries whose effective candidate set is smaller than --candidate-size. "
                        "By default the script fails when the candidate pool degenerates, because "
                        "Top-k metrics become misleading (e.g. candidate size 1 gives Top1=100%).")
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


def _load_epi_pairs(
    args, task_key: str
) -> tuple[list[RetrievalExample], list[str], dict[str, set[str]], dict[str, set[str]]]:
    """
    Returns:
        positives           : list of (enhancer, true_promoter) RetrievalExamples
                              from rows where label == 1.
        unique_promoters    : sorted unique list of every promoter seen in the
                              split (used as the global candidate pool).
        per_enhancer_pool   : dict[enhancer -> set[promoter]] of every promoter
                              that appeared on a row with that enhancer (used
                              when --candidate-pool=per-enhancer).
        positive_targets    : dict[enhancer -> set[promoter]] of every positive
                              promoter in the full split. This is intentionally
                              computed before --max-queries truncation so quick
                              pilots do not accidentally sample held-out true
                              positives as negatives.
    """
    subdir = base._EPI_SUBDIR_DEFAULT.get(task_key, task_key)
    task_dir = os.path.join(args.gue_plus_dir, "EPI", subdir)
    path = os.path.join(task_dir, f"{args.split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing EPI split file: {path}")

    positives: list[RetrievalExample] = []
    promoter_pool: list[str] = []
    per_enhancer_pool: dict[str, set[str]] = defaultdict(set)
    positive_targets: dict[str, set[str]] = defaultdict(set)
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

            try:
                label = int(row["label"])
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(
                    f"{path} row {row_idx}: invalid label {row.get('label')!r}; "
                    f"expected an integer (0 or 1)."
                ) from e

            promoter_pool.append(promoter)
            per_enhancer_pool[enhancer].add(promoter)
            if label == 1:
                positive_targets[enhancer].add(promoter)
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
    return positives, unique_promoters, dict(per_enhancer_pool), dict(positive_targets)


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
    """
    Return the 1-based rank of `true_idx` among `candidate_indices`, scored by
    cosine similarity (dot product on L2-normalised embeddings).

    Tie-breaking: ties with the true target are resolved by the (already
    randomly-shuffled) position of the true target in `candidate_indices`,
    counting only strictly-higher scores plus equal scores at earlier
    positions. Since the candidate order is randomised by the caller, this
    gives the expected mid-rank under random tie-break.
    """
    scores = promoter_embeddings[candidate_indices] @ query
    true_positions = np.flatnonzero(candidate_indices == true_idx)
    if true_positions.size != 1:
        raise ValueError("Candidate set must contain the true promoter exactly once.")
    true_pos = int(true_positions[0])
    true_score = float(scores[true_pos])
    higher = int(np.sum(scores > true_score))
    equal_before = int(np.sum((scores[:true_pos] == true_score)))
    return higher + equal_before + 1


def _expected_random_metrics(candidate_sizes: list[int]) -> dict[str, float]:
    """Expected retrieval metrics under a uniformly random ranking.

    Each query has exactly one relevant promoter. Candidate-set sizes can vary
    when the requested pool is larger than the available negatives, so compute
    the random baseline per query and average.
    """
    sizes = np.asarray(candidate_sizes, dtype=np.float64)
    if sizes.size == 0:
        return {
            "random_top1": float("nan"),
            "random_top5": float("nan"),
            "random_top10": float("nan"),
            "random_mrr": float("nan"),
            "random_median_rank": float("nan"),
            "random_mean_rank": float("nan"),
        }

    def random_topk(k: int) -> float:
        return float(np.mean(np.minimum(k, sizes) / sizes))

    # Expected reciprocal rank for one relevant item uniformly placed in 1..N:
    # H_N / N. Candidate sets are usually <=1000, so the direct sum is fine.
    random_mrr = float(np.mean([np.sum(1.0 / np.arange(1, int(n) + 1)) / n for n in sizes]))
    return {
        "random_top1": random_topk(1),
        "random_top5": random_topk(5),
        "random_top10": random_topk(10),
        "random_mrr": random_mrr,
        "random_median_rank": float(np.median((sizes + 1.0) / 2.0)),
        "random_mean_rank": float(np.mean((sizes + 1.0) / 2.0)),
    }


def _evaluate_retrieval_for_model(
    model_name: str,
    query_embeddings: np.ndarray,
    promoter_embeddings: np.ndarray,
    examples: list[RetrievalExample],
    promoter_to_idx: dict[str, int],
    promoter_pool: list[str],
    per_enhancer_pool: dict[str, set[str]],
    positive_targets_by_enhancer: dict[str, set[str]],
    candidate_pool_mode: str,
    candidate_size: int,
    allow_short_candidates: bool,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """
    Evaluate enhancer-to-promoter retrieval for one model on one task.

    `query_embeddings[q_idx]` must be the embedding for `examples[q_idx].enhancer`.
    The negative pool for each query depends on `candidate_pool_mode`:
      - "global"       : sample from all promoters in the split (minus this
                         enhancer's known targets).
      - "per-enhancer" : sample only from promoters that appear on this
                         enhancer's rows in the data file (minus this
                         enhancer's known true targets).
    """
    rng = np.random.default_rng(seed)
    known_targets = positive_targets_by_enhancer or _known_targets_by_enhancer(examples)
    global_pool_indices = np.asarray(
        [promoter_to_idx[p] for p in promoter_pool], dtype=np.int64
    )

    # Pre-compute the negative pool per enhancer once (reused across repeats).
    enh_to_negpool: dict[str, np.ndarray] = {}
    for enh in {ex.enhancer for ex in examples}:
        forbidden_idx = np.asarray(
            [promoter_to_idx[p] for p in known_targets.get(enh, set())],
            dtype=np.int64,
        )
        if candidate_pool_mode == "per-enhancer":
            base_pool = np.asarray(
                [promoter_to_idx[p] for p in per_enhancer_pool.get(enh, set())],
                dtype=np.int64,
            )
        else:  # "global"
            base_pool = global_pool_indices
        enh_to_negpool[enh] = (
            np.setdiff1d(base_pool, forbidden_idx, assume_unique=False).astype(np.int64)
        )

    all_repeat_metrics: list[dict[str, float]] = []
    repeat_ranks: list[np.ndarray] = []
    query_records: list[dict[str, Any]] = []
    effective_candidate_sizes: list[int] = []
    short_pool_warned = False

    for repeat in range(repeats):
        ranks = []
        for q_idx, ex in enumerate(
            tqdm(examples, desc=f"{model_name} retrieval r{repeat + 1}", leave=False)
        ):
            true_idx = promoter_to_idx[ex.promoter]
            negative_pool = enh_to_negpool[ex.enhancer]
            n_neg = max(0, min(candidate_size - 1, len(negative_pool)))
            if n_neg:
                sampled = rng.choice(negative_pool, size=n_neg, replace=False)
                candidate_indices = np.concatenate(
                    [np.asarray([true_idx], dtype=np.int64), sampled]
                )
            else:
                candidate_indices = np.asarray([true_idx], dtype=np.int64)
            rng.shuffle(candidate_indices)

            if len(candidate_indices) < candidate_size:
                message = (
                    f"{model_name}: enhancer at row {ex.row_idx} produced only "
                    f"{len(candidate_indices)} candidates (requested {candidate_size}). "
                    f"This usually means --candidate-pool={candidate_pool_mode!r} is "
                    f"degenerate for this split."
                )
                if not allow_short_candidates:
                    raise ValueError(
                        message
                        + " Use --candidate-pool global, lower --candidate-size, or pass "
                        "--allow-short-candidates only for debugging."
                    )
                if not short_pool_warned:
                    print(f"  [warn] {message}")
                    short_pool_warned = True

            rank = _rank_true_target(
                query_embeddings[q_idx],
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
        repeat_ranks.append(ranks_arr)
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
    random_metrics = _expected_random_metrics(effective_candidate_sizes)
    summary.update(random_metrics)
    for key in ("top1", "top5", "top10", "mrr"):
        summary[f"{key}_minus_random"] = summary[key] - summary[f"random_{key}"]
    summary["n_queries"] = len(examples)
    summary["candidate_pool"] = candidate_pool_mode
    summary["candidate_size_requested"] = candidate_size
    summary["candidate_size_effective_mean"] = float(np.mean(effective_candidate_sizes))
    summary["candidate_size_effective_median"] = float(np.median(effective_candidate_sizes))
    summary["candidate_size_effective_min"] = int(np.min(effective_candidate_sizes))
    summary["repeats"] = repeats
    return {
        "summary": summary,
        "repeat_metrics": all_repeat_metrics,
        "queries_first_repeat": query_records,
        # Raw ranks from the first repeat, used to pool across tasks for a true
        # cross-task median (linear-aggregation of medians is misleading).
        "ranks_first_repeat": repeat_ranks[0].tolist(),
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
        "candidate_size_effective_median",
        "top1",
        "top5",
        "top10",
        "mrr",
        "random_top1",
        "random_top5",
        "random_top10",
        "random_mrr",
        "top1_minus_random",
        "top5_minus_random",
        "top10_minus_random",
        "mrr_minus_random",
        "median_rank",
        "mean_rank",
        "random_median_rank",
        "random_mean_rank",
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
    print(f"  Models         : {[s.name for s in specs]}")
    print(f"  Tasks          : {task_keys}")
    print(f"  Split          : {args.split}")
    print(f"  Candidate pool : {args.candidate_pool}")
    print(f"  Candidates     : {args.candidate_size} (including true promoter)")
    print(f"  Repeats        : {args.repeats}")
    print(f"  Pooling        : {args.pooling}")
    print(f"  Device         : {device}")
    print("=" * 72)

    task_payloads = {}
    for task_key in task_keys:
        examples, promoter_pool, per_enhancer_pool, positive_targets = _load_epi_pairs(args, task_key)
        task_payloads[task_key] = (examples, promoter_pool, per_enhancer_pool, positive_targets)
        print(
            f"  {task_key:<5} {examples[0].task_name:<12} "
            f"queries={len(examples):>5} promoters={len(promoter_pool):>5} "
            f"enhancers_w_pool={len(per_enhancer_pool):>5}"
        )
        if args.candidate_pool == "per-enhancer":
            pool_sizes = np.asarray([len(v) for v in per_enhancer_pool.values()], dtype=np.int64)
            print(
                f"        per-enhancer pool sizes: "
                f"mean={pool_sizes.mean():.2f}, median={np.median(pool_sizes):.1f}, "
                f"max={pool_sizes.max()}"
            )

    results: dict[str, Any] = {
        "config": vars(args),
        "tasks": {},
        "models": [spec.__dict__ for spec in specs],
    }
    summary_rows: list[dict[str, Any]] = []
    # ranks_first_repeat per (spec.name, task_key), used for cross-task median pooling
    ranks_by_model: dict[str, list[float]] = defaultdict(list)

    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer = base.load_model(spec, device, dtype)

        for task_key in task_keys:
            examples, promoter_pool, per_enhancer_pool, positive_targets = task_payloads[task_key]
            unique_promoters = promoter_pool
            promoter_to_idx = {promoter: i for i, promoter in enumerate(unique_promoters)}

            # Deduplicate enhancers before encoding (saves GPU time when the same
            # enhancer appears in multiple positive rows). We keep an index map
            # so we can look up the embedding by enhancer string later.
            unique_enhancer_list = sorted({ex.enhancer for ex in examples})
            enh_to_emb_idx = {e: i for i, e in enumerate(unique_enhancer_list)}

            unique_enhancer_embeddings = base.encode_sequences(
                model,
                tokenizer,
                unique_enhancer_list,
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

            # Build query-embedding-per-example by indexing into the deduplicated table.
            query_embeddings = np.stack(
                [unique_enhancer_embeddings[enh_to_emb_idx[ex.enhancer]] for ex in examples]
            )

            model_result = _evaluate_retrieval_for_model(
                model_name=spec.name,
                query_embeddings=query_embeddings,
                promoter_embeddings=promoter_embeddings,
                examples=examples,
                promoter_to_idx=promoter_to_idx,
                promoter_pool=promoter_pool,
                per_enhancer_pool=per_enhancer_pool,
                positive_targets_by_enhancer=positive_targets,
                candidate_pool_mode=args.candidate_pool,
                candidate_size=args.candidate_size,
                allow_short_candidates=args.allow_short_candidates,
                repeats=args.repeats,
                seed=args.seed,
            )
            summary = model_result["summary"]
            print(
                f"   {task_key:<5} Top1={summary['top1']*100:6.2f} "
                f"Top10={summary['top10']*100:6.2f} "
                f"(rand {summary['random_top10']*100:5.2f}) "
                f"MRR={summary['mrr']:.4f} "
                f"(rand {summary['random_mrr']:.4f}) "
                f"median_rank={summary['median_rank']:.1f}"
            )

            results["tasks"].setdefault(task_key, {})[spec.name] = model_result
            ranks_by_model[spec.name].extend(model_result["ranks_first_repeat"])
            summary_rows.append(
                {
                    "model": spec.name,
                    "task_key": task_key,
                    "task_name": examples[0].task_name,
                    **summary,
                }
            )

            del unique_enhancer_embeddings, query_embeddings, promoter_embeddings
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
                np.average(
                    [row["candidate_size_effective_mean"] for row in model_rows],
                    weights=weights,
                )
            ),
            "candidate_size_effective_median": float(
                np.median([row["candidate_size_effective_median"] for row in model_rows])
            ),
        }
        # top-K, mrr, mean_rank are linear in queries, so weighted average is correct.
        for key in (
            "top1",
            "top5",
            "top10",
            "mrr",
            "mean_rank",
            "random_top1",
            "random_top5",
            "random_top10",
            "random_mrr",
            "random_mean_rank",
            "top1_minus_random",
            "top5_minus_random",
            "top10_minus_random",
            "mrr_minus_random",
        ):
            aggregate[key] = float(np.average([row[key] for row in model_rows], weights=weights))
        # median_rank is non-linear; pool the raw ranks across tasks (first repeat only).
        pooled_ranks = np.asarray(ranks_by_model[spec.name], dtype=np.float64)
        aggregate["median_rank"] = float(np.median(pooled_ranks)) if pooled_ranks.size else float("nan")
        aggregate["random_median_rank"] = float(
            np.median([row["random_median_rank"] for row in model_rows])
        )
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
