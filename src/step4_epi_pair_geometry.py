"""
Step 4 add-on: EPI pair-representation geometry.

This script evaluates whether pair embeddings encode EPI labels without
training a full linear classifier. It uses:
  1. positive/negative prototype scoring,
  2. kNN label scoring in embedding space, and
  3. a lightweight positive-vs-negative pair-ranking diagnostic.

Unlike enhancer-to-promoter retrieval, each item is the full
enhancer+promoter pair, matching the original EPI classification input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import step4_evaluate as base  # noqa: E402


@dataclass
class PairDataset:
    task_key: str
    task_name: str
    split: str
    sequences: list[str]
    labels: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description="Run EPI pair-embedding separability diagnostics.")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode")
    p.add_argument("--gue-plus-dir", required=True, help="Root directory containing EPI/")
    p.add_argument(
        "--task-key",
        default="all",
        choices=["all"] + [f"epi_{i}" for i in range(6)],
        help="Which EPI task to evaluate, or all six tasks.",
    )
    p.add_argument("--train-split", default="train", choices=("train", "dev", "test"))
    p.add_argument("--eval-split", default="dev", choices=("train", "dev", "test"))
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--max-eval-pairs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"),
                   default="mean")
    p.add_argument("--epi-crop-bp", type=int, default=4096)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--filter-n", action="store_true")
    p.add_argument("--knn-k", type=int, default=25)
    p.add_argument("--ranking-candidate-size", type=int, default=100,
                   help="1 positive + candidate_size-1 negative eval pairs for pair-ranking.")
    p.add_argument("--ranking-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-prefix", required=True)
    return p.parse_args()


def _task_display_name(task_key: str) -> str:
    for key, _n_classes, display_name, _source in base.GUE_PLUS_EPI_BENCHMARKS:
        if key == task_key:
            return display_name
    return task_key


def _load_pair_dataset(args, task_key: str, split: str, cap: int | None) -> PairDataset:
    subdir = base._EPI_SUBDIR_DEFAULT.get(task_key, task_key)
    task_dir = os.path.join(args.gue_plus_dir, "EPI", subdir)
    path = os.path.join(task_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing EPI split file: {path}")

    sequences: list[str] = []
    labels: list[int] = []
    dropped = 0

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"enhancer", "promoter", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} must contain enhancer/promoter/label columns; missing {sorted(missing)}"
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
                raise ValueError(f"{path} row {row_idx}: invalid label {row.get('label')!r}") from e
            if label not in (0, 1):
                raise ValueError(f"{path} row {row_idx}: label must be 0/1, got {label!r}")

            seq = enhancer + promoter
            anchor = len(enhancer)
            seq = base._crop_sequence(seq, args.epi_crop_bp, args.epi_crop_mode, anchor)
            if args.filter_n and not base._is_acgt(seq):
                dropped += 1
                continue
            sequences.append(seq)
            labels.append(label)

    if cap is not None:
        sequences = sequences[:cap]
        labels = labels[:cap]

    if dropped:
        print(f"  {task_key}/{split}: dropped {dropped} rows with empty/non-ACGT regions")
    if not sequences:
        raise ValueError(f"No EPI pairs loaded for {task_key}/{split}.")

    return PairDataset(
        task_key=task_key,
        task_name=_task_display_name(task_key),
        split=split,
        sequences=sequences,
        labels=np.asarray(labels, dtype=np.int64),
    )


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan"), float("nan")


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    pred = (scores >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def _best_mcc_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef

    candidates = np.unique(scores)
    if candidates.size > 512:
        candidates = np.quantile(scores, np.linspace(0.01, 0.99, 512))
    best_threshold = float(candidates[0])
    best_mcc = -2.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = float(threshold)
    return best_threshold


def _prototype_scores(train_emb: np.ndarray, train_y: np.ndarray, eval_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.sum(train_y == 1) == 0 or np.sum(train_y == 0) == 0:
        raise ValueError("Prototype scoring requires both positive and negative train labels.")
    pos = train_emb[train_y == 1].mean(axis=0)
    neg = train_emb[train_y == 0].mean(axis=0)
    prototypes = _normalise_rows(np.stack([neg, pos]))
    train_scores = train_emb @ prototypes[1] - train_emb @ prototypes[0]
    eval_scores = eval_emb @ prototypes[1] - eval_emb @ prototypes[0]
    return train_scores, eval_scores


def _knn_scores(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    eval_emb: np.ndarray,
    k: int,
    batch_size: int = 512,
) -> np.ndarray:
    k_eff = max(1, min(k, len(train_y)))
    scores = []
    for start in range(0, eval_emb.shape[0], batch_size):
        batch = eval_emb[start : start + batch_size]
        sim = batch @ train_emb.T
        top_idx = np.argpartition(sim, kth=-k_eff, axis=1)[:, -k_eff:]
        scores.append(train_y[top_idx].mean(axis=1))
    return np.concatenate(scores, axis=0)


def _ranking_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    candidate_size: int,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {
            "pair_rank_top1": float("nan"),
            "pair_rank_top10": float("nan"),
            "pair_rank_mrr": float("nan"),
            "pair_rank_median_rank": float("nan"),
        }

    n_neg = min(candidate_size - 1, neg_idx.size)
    all_ranks = []
    for _ in range(repeats):
        for pidx in pos_idx:
            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            cand = np.concatenate([np.asarray([pidx]), sampled_neg])
            rng.shuffle(cand)
            cand_scores = scores[cand]
            true_pos = int(np.flatnonzero(cand == pidx)[0])
            true_score = float(cand_scores[true_pos])
            rank = int(np.sum(cand_scores > true_score) + np.sum(cand_scores[:true_pos] == true_score) + 1)
            all_ranks.append(rank)

    ranks = np.asarray(all_ranks, dtype=np.float64)
    return {
        "pair_rank_top1": float(np.mean(ranks <= 1)),
        "pair_rank_top10": float(np.mean(ranks <= 10)),
        "pair_rank_mrr": float(np.mean(1.0 / ranks)),
        "pair_rank_median_rank": float(np.median(ranks)),
    }


def _evaluate_geometry(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    eval_emb: np.ndarray,
    eval_y: np.ndarray,
    knn_k: int,
    ranking_candidate_size: int,
    ranking_repeats: int,
    seed: int,
) -> dict[str, Any]:
    train_emb = _normalise_rows(train_emb)
    eval_emb = _normalise_rows(eval_emb)

    proto_train_scores, proto_eval_scores = _prototype_scores(train_emb, train_y, eval_emb)
    proto_threshold = _best_mcc_threshold(train_y, proto_train_scores)
    proto_auc, proto_ap = _safe_auc(eval_y, proto_eval_scores)
    proto_metrics = {
        "prototype_auc": proto_auc,
        "prototype_ap": proto_ap,
        "prototype_threshold": proto_threshold,
        **{f"prototype_{k}": v for k, v in _binary_metrics(eval_y, proto_eval_scores, proto_threshold).items()},
        **_ranking_metrics(
            proto_eval_scores,
            eval_y,
            candidate_size=ranking_candidate_size,
            repeats=ranking_repeats,
            seed=seed,
        ),
    }

    knn_train_scores = _knn_scores(train_emb, train_y, train_emb, k=knn_k)
    knn_eval_scores = _knn_scores(train_emb, train_y, eval_emb, k=knn_k)
    # kNN scores are neighbour positive-label fractions, so 0.5 is the natural
    # majority-vote threshold. Avoid tuning this on train because train kNN
    # scores include self-neighbours.
    knn_threshold = 0.5
    knn_auc, knn_ap = _safe_auc(eval_y, knn_eval_scores)
    knn_metrics = {
        "knn_k": knn_k,
        "knn_auc": knn_auc,
        "knn_ap": knn_ap,
        "knn_threshold": knn_threshold,
        **{f"knn_{k}": v for k, v in _binary_metrics(eval_y, knn_eval_scores, knn_threshold).items()},
    }

    return {
        **proto_metrics,
        **knn_metrics,
        "n_train": int(len(train_y)),
        "n_eval": int(len(eval_y)),
        "train_pos_rate": float(np.mean(train_y)),
        "eval_pos_rate": float(np.mean(eval_y)),
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = [base.ModelSpec.parse(s) for s in args.models]
    task_keys = [f"epi_{i}" for i in range(6)] if args.task_key == "all" else [args.task_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 72)
    print("Step 4 add-on | EPI pair-representation geometry")
    print("=" * 72)
    print(f"  Models       : {[s.name for s in specs]}")
    print(f"  Tasks        : {task_keys}")
    print(f"  Train/eval   : {args.train_split} -> {args.eval_split}")
    print(f"  Crop         : {args.epi_crop_bp} bp ({args.epi_crop_mode})")
    print(f"  Pooling      : {args.pooling}")
    print(f"  kNN k        : {args.knn_k}")
    print(f"  Device       : {device}")
    print("=" * 72)

    datasets: dict[str, tuple[PairDataset, PairDataset]] = {}
    for task_key in task_keys:
        train_ds = _load_pair_dataset(args, task_key, args.train_split, args.max_train_pairs)
        eval_ds = _load_pair_dataset(args, task_key, args.eval_split, args.max_eval_pairs)
        datasets[task_key] = (train_ds, eval_ds)
        print(
            f"  {task_key:<5} {train_ds.task_name:<12} "
            f"train={len(train_ds.labels):>5} eval={len(eval_ds.labels):>5} "
            f"train_pos={train_ds.labels.mean():.3f} eval_pos={eval_ds.labels.mean():.3f}"
        )

    results: dict[str, Any] = {"config": vars(args), "models": [spec.__dict__ for spec in specs], "tasks": {}}
    rows: list[dict[str, Any]] = []

    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer = base.load_model(spec, device, dtype)

        for task_key in task_keys:
            train_ds, eval_ds = datasets[task_key]
            train_emb = base.encode_sequences(
                model,
                tokenizer,
                train_ds.sequences,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
                pooling=args.pooling,
                desc=f"{spec.name} {task_key} {args.train_split}",
            )
            eval_emb = base.encode_sequences(
                model,
                tokenizer,
                eval_ds.sequences,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
                pooling=args.pooling,
                desc=f"{spec.name} {task_key} {args.eval_split}",
            )

            metrics = _evaluate_geometry(
                train_emb=train_emb,
                train_y=train_ds.labels,
                eval_emb=eval_emb,
                eval_y=eval_ds.labels,
                knn_k=args.knn_k,
                ranking_candidate_size=args.ranking_candidate_size,
                ranking_repeats=args.ranking_repeats,
                seed=args.seed,
            )
            results["tasks"].setdefault(task_key, {})[spec.name] = metrics
            row = {"model": spec.name, "task_key": task_key, "task_name": train_ds.task_name, **metrics}
            rows.append(row)

            print(
                f"   {task_key:<5} proto_auc={metrics['prototype_auc']:.4f} "
                f"proto_mcc={metrics['prototype_mcc']:.4f} "
                f"knn_auc={metrics['knn_auc']:.4f} knn_mcc={metrics['knn_mcc']:.4f} "
                f"rankTop10={metrics['pair_rank_top10']*100:.2f}"
            )

            del train_emb, eval_emb
            if device == "cuda":
                torch.cuda.empty_cache()

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    for spec in specs:
        model_rows = [row for row in rows if row["model"] == spec.name]
        if len(model_rows) <= 1:
            continue
        weights = np.asarray([row["n_eval"] for row in model_rows], dtype=np.float64)
        aggregate = {"model": spec.name, "task_key": "all", "task_name": "EPI-Avg"}
        metric_keys = [key for key in model_rows[0] if key not in {"model", "task_key", "task_name"}]
        for key in metric_keys:
            vals = np.asarray([row[key] for row in model_rows], dtype=np.float64)
            if np.all(np.isfinite(vals)):
                aggregate[key] = float(np.average(vals, weights=weights))
        rows.append(aggregate)
        results.setdefault("overall", {})[spec.name] = aggregate

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    with open(f"{args.output_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _write_csv(f"{args.output_prefix}.csv", rows)
    print("\nSaved:")
    print(f"  JSON summary: {args.output_prefix}.json")
    print(f"  CSV summary : {args.output_prefix}.csv")


if __name__ == "__main__":
    main()
