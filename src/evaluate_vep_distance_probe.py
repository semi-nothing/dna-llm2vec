#!/usr/bin/env python
"""
Caduceus-style frozen VEP/SNP distance-bucket probe.

VEP labels describe the effect of a mutation, so the probe feature is built
from paired reference and alternate windows:
  - diff       : emb_alt - emb_ref
  - absdiff    : |emb_alt - emb_ref|
  - concat     : [emb_ref; emb_alt; |emb_alt - emb_ref|]

The script reports both linear and RBF-SVM probes by default. Linear is the
cleaner representation-quality metric; RBF-SVM is included for closer
comparison with Caduceus-style protocols. Distance buckets are evaluated
separately, and paired bootstrap deltas are computed against the first model
or --baseline-model using identical SNP test subsets.

Input CSV columns default to:
  ref_sequence, alt_sequence, label, distance_to_tss

Example:
  uv run python src/evaluate_vep_distance_probe.py \
    --models "H0:LongSafari/hyenadna-small-32k-seqlen-hf:hyena" \
             "H1:./hyena_bidir_h1:hyena" \
    --csv ./data/vep_windows.csv \
    --feature diff \
    --probes linear rbf_svm \
    --train-per-bucket 5000 \
    --repeats 5 \
    --output ./eval_results/vep_distance_probe.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
HYENA_DIR = os.path.join(ROOT, "src_hyena")
for path in (ROOT, SRC_DIR, HYENA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import step4_evaluate as base  # noqa: E402
import step4_hyena_evaluate as hyena_eval  # noqa: E402


@dataclass
class ModelSpec:
    name: str
    path: str
    mode: str

    @staticmethod
    def parse(spec: str) -> "ModelSpec":
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad model spec {spec!r}; expected NAME:PATH:MODE, "
                "where MODE is causal, bidir, encoder, or hyena."
            )
        name, path, mode = parts
        if mode not in {"causal", "bidir", "encoder", "hyena"}:
            raise ValueError(f"Unsupported mode {mode!r}.")
        return ModelSpec(name=name, path=path, mode=mode)


def load_model(spec: ModelSpec, device: str, dtype: torch.dtype):
    if spec.mode == "hyena":
        hyena_spec = base.ModelSpec(name=spec.name, path=spec.path, mode="encoder")
        model, tokenizer = hyena_eval.load_model(hyena_spec, device=device, dtype=dtype)
        return model, tokenizer, hyena_eval.encode_sequences

    generic_spec = base.ModelSpec(name=spec.name, path=spec.path, mode=spec.mode)
    model, tokenizer = base.load_model(generic_spec, device=device, dtype=dtype)
    return model, tokenizer, base.encode_sequences


def encode_sequences(
    encode_fn,
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str,
    normalize: bool,
    desc: str,
) -> np.ndarray:
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        pooling=pooling,
        desc=desc,
    )
    if encode_fn is base.encode_sequences:
        kwargs["normalize"] = normalize
    emb = encode_fn(**kwargs)
    if normalize and encode_fn is not base.encode_sequences:
        denom = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(denom, 1e-12, None)
    return emb.astype(np.float32, copy=False)


def parse_bucket_edges(text: str) -> list[float]:
    edges = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(edges) < 2:
        raise ValueError("--distance-bins must contain at least two comma-separated numbers.")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--distance-bins must be strictly increasing.")
    return edges


def bucket_name(lo: float, hi: float) -> str:
    def fmt(x: float) -> str:
        if x >= 1_000_000:
            return f"{x / 1_000_000:g}Mb"
        if x >= 1_000:
            return f"{x / 1_000:g}kb"
        return f"{x:g}bp"

    return f"{fmt(lo)}-{fmt(hi)}"


def read_rows(args) -> list[dict]:
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {args.ref_sequence_col, args.alt_sequence_col, args.label_col, args.distance_col}
        if args.split_col:
            required.add(args.split_col)
        missing = required - fieldnames
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            ref_seq = (row.get(args.ref_sequence_col) or "").strip().upper()
            alt_seq = (row.get(args.alt_sequence_col) or "").strip().upper()
            if not ref_seq or not alt_seq or len(ref_seq) != len(alt_seq):
                continue
            try:
                label = int(float(row[args.label_col]))
                distance = abs(float(row[args.distance_col]))
            except ValueError:
                continue
            split = (row.get(args.split_col, "") if args.split_col else "").strip().lower()
            rows.append(
                {
                    "index": row_idx,
                    "ref_sequence": ref_seq,
                    "alt_sequence": alt_seq,
                    "label": label,
                    "distance": distance,
                    "split": split,
                }
            )
    if not rows:
        raise ValueError("No usable paired ref/alt rows found in the VEP CSV.")
    return rows


def assign_buckets(rows: list[dict], edges: list[float]) -> dict[str, list[int]]:
    buckets = {bucket_name(lo, hi): [] for lo, hi in zip(edges, edges[1:])}
    for idx, row in enumerate(rows):
        dist = row["distance"]
        for lo, hi in zip(edges, edges[1:]):
            if lo <= dist < hi:
                buckets[bucket_name(lo, hi)].append(idx)
                break
    return buckets


def make_split(
    candidate_indices: list[int],
    rows: list[dict],
    rng: random.Random,
    train_per_bucket: int,
    test_per_bucket: int,
    split_col: bool,
    train_value: str,
    test_value: str,
    test_fraction: float,
) -> tuple[list[int], list[int]]:
    if split_col:
        train_idx = [idx for idx in candidate_indices if rows[idx]["split"] == train_value]
        test_idx = [idx for idx in candidate_indices if rows[idx]["split"] == test_value]
    else:
        shuffled = list(candidate_indices)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        test_idx = shuffled[:n_test]
        train_idx = shuffled[n_test:]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    if train_per_bucket > 0:
        train_idx = train_idx[:train_per_bucket]
    if test_per_bucket > 0:
        test_idx = test_idx[:test_per_bucket]
    return train_idx, test_idx


def build_variant_features(ref_emb: np.ndarray, alt_emb: np.ndarray, mode: str) -> np.ndarray:
    delta = alt_emb - ref_emb
    if mode == "diff":
        return delta
    if mode == "absdiff":
        return np.abs(delta)
    if mode == "concat":
        return np.concatenate([ref_emb, alt_emb, np.abs(delta)], axis=1)
    raise ValueError(f"Unsupported feature mode {mode!r}.")


def fit_probe(train_x, train_y, test_x, probe: str, seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if probe == "linear":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(train_x, train_y)
        return pipe.predict_proba(test_x)[:, 1]

    from sklearn.svm import SVC

    clf = SVC(kernel="rbf", class_weight="balanced", gamma="scale", C=1.0, random_state=seed)
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(train_x, train_y)
    return pipe.decision_function(test_x)


def auc_or_nan(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(np.asarray(y_true).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_delta(y, base_scores, scores, rng, n_iters: int) -> dict:
    deltas = []
    y = np.asarray(y)
    base_scores = np.asarray(base_scores)
    scores = np.asarray(scores)
    valid = np.isfinite(base_scores) & np.isfinite(scores)
    y = y[valid]
    base_scores = base_scores[valid]
    scores = scores[valid]
    if y.size < 2 or len(set(y.tolist())) < 2:
        return {"delta_auc_mean": float("nan"), "delta_auc_ci95": [float("nan"), float("nan")], "n_bootstrap": 0}

    for _ in range(n_iters):
        sample = rng.integers(0, y.size, size=y.size)
        y_s = y[sample]
        if len(set(y_s.tolist())) < 2:
            continue
        deltas.append(auc_or_nan(y_s, scores[sample]) - auc_or_nan(y_s, base_scores[sample]))

    finite = np.array([x for x in deltas if np.isfinite(x)], dtype=np.float64)
    return {
        "delta_auc_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "delta_auc_ci95": [
            float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
            float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        ],
        "n_bootstrap": int(finite.size),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run frozen VEP/SNP distance-bucket probe.")
    p.add_argument("--models", nargs="+", required=True, help="NAME:PATH:MODE specs; MODE includes hyena.")
    p.add_argument("--csv", required=True, help="CSV containing paired SNP-centered ref/alt windows and labels.")
    p.add_argument("--ref-sequence-col", default="ref_sequence")
    p.add_argument("--alt-sequence-col", default="alt_sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--distance-col", default="distance_to_tss")
    p.add_argument("--split-col", default=None, help="Optional column with train/test split labels.")
    p.add_argument("--train-value", default="train")
    p.add_argument("--test-value", default="test")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--distance-bins", default="0,10000,100000,1000000000")
    p.add_argument("--train-per-bucket", type=int, default=5000)
    p.add_argument("--test-per-bucket", type=int, default=0, help="0 keeps all available test rows.")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--feature", choices=("diff", "absdiff", "concat"), default="diff")
    p.add_argument("--probes", nargs="+", choices=("linear", "rbf_svm"), default=["linear", "rbf_svm"])
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--baseline-model", default=None, help="Model name for paired bootstrap deltas. Defaults to first model.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hyena-batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--hyena-max-length", type=int, default=8192)
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--output", default="./eval_results/vep_distance_probe.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.train_value = args.train_value.lower()
    args.test_value = args.test_value.lower()
    rows = read_rows(args)
    edges = parse_bucket_edges(args.distance_bins)
    buckets = assign_buckets(rows, edges)
    specs = [ModelSpec.parse(item) for item in args.models]
    baseline_name = args.baseline_model or specs[0].name

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.fp32 or device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("VEP distance-bucket frozen probe")
    print(f"  Rows      : {len(rows):,}")
    print(f"  Buckets   : {', '.join(f'{k}={len(v):,}' for k, v in buckets.items())}")
    print(f"  Feature   : {args.feature}")
    print(f"  Probes    : {args.probes}")
    print(f"  Baseline  : {baseline_name}")
    print(f"  Repeats   : {args.repeats}")
    print(f"  Device    : {device}")

    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    ref_sequences = [row["ref_sequence"] for row in rows]
    alt_sequences = [row["alt_sequence"] for row in rows]
    results = {
        "config": vars(args),
        "n_rows": len(rows),
        "bucket_counts": {name: len(indices) for name, indices in buckets.items()},
        "models": {},
        "paired_bootstrap": {},
    }
    score_cache = {}

    for spec in specs:
        print(f"\n-> {spec.name}")
        model, tokenizer, encode_fn = load_model(spec, device, dtype)
        if spec.mode == "hyena":
            batch_size = args.hyena_batch_size or args.batch_size
        else:
            batch_size = args.batch_size
        max_length = args.hyena_max_length if spec.mode == "hyena" else args.max_length

        ref_embeddings = encode_sequences(
            encode_fn,
            model,
            tokenizer,
            ref_sequences,
            batch_size,
            max_length,
            device,
            args.pooling,
            normalize=not args.no_l2_normalize,
            desc=f"{spec.name} VEP ref windows",
        )
        alt_embeddings = encode_sequences(
            encode_fn,
            model,
            tokenizer,
            alt_sequences,
            batch_size,
            max_length,
            device,
            args.pooling,
            normalize=not args.no_l2_normalize,
            desc=f"{spec.name} VEP alt windows",
        )
        features = build_variant_features(ref_embeddings, alt_embeddings, args.feature)

        model_result = {"model": spec.__dict__, "feature_dim": int(features.shape[1]), "buckets": {}}
        for bucket, indices in buckets.items():
            model_result["buckets"][bucket] = {}
            for probe in args.probes:
                aucs = []
                repeat_details = []
                for repeat in range(args.repeats):
                    rng = random.Random(args.seed + repeat)
                    train_idx, test_idx = make_split(
                        candidate_indices=indices,
                        rows=rows,
                        rng=rng,
                        train_per_bucket=args.train_per_bucket,
                        test_per_bucket=args.test_per_bucket,
                        split_col=args.split_col is not None,
                        train_value=args.train_value,
                        test_value=args.test_value,
                        test_fraction=args.test_fraction,
                    )
                    train_y = labels[train_idx]
                    test_y = labels[test_idx]
                    if len(train_idx) < 2 or len(test_idx) < 2 or len(set(train_y.tolist())) < 2:
                        scores = np.full(len(test_idx), np.nan, dtype=np.float64)
                        auc = float("nan")
                    else:
                        scores = fit_probe(
                            features[train_idx],
                            train_y,
                            features[test_idx],
                            probe=probe,
                            seed=args.seed + repeat,
                        )
                        auc = auc_or_nan(test_y, scores)
                    aucs.append(auc)
                    repeat_details.append(
                        {
                            "repeat": repeat,
                            "train_n": len(train_idx),
                            "test_n": len(test_idx),
                            "test_indices": [int(idx) for idx in test_idx],
                            "aucroc": auc,
                        }
                    )
                    score_cache[(spec.name, bucket, probe, repeat)] = {
                        "indices": np.array(test_idx, dtype=np.int64),
                        "labels": test_y.astype(np.int64, copy=False),
                        "scores": np.asarray(scores, dtype=np.float64),
                    }

                finite = np.array([x for x in aucs if np.isfinite(x)], dtype=np.float64)
                model_result["buckets"][bucket][probe] = {
                    "aucroc_mean": float(np.mean(finite)) if finite.size else float("nan"),
                    "aucroc_std": float(np.std(finite)) if finite.size else float("nan"),
                    "repeats": repeat_details,
                }
                print(
                    f"  {bucket:<18} {probe:<7} AUC={model_result['buckets'][bucket][probe]['aucroc_mean']:.4f} "
                    f"+/- {model_result['buckets'][bucket][probe]['aucroc_std']:.4f}"
                )

        results["models"][spec.name] = model_result
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    for spec in specs:
        if spec.name == baseline_name:
            continue
        results["paired_bootstrap"][spec.name] = {}
        for bucket in buckets:
            results["paired_bootstrap"][spec.name][bucket] = {}
            for probe in args.probes:
                per_repeat = []
                for repeat in range(args.repeats):
                    base_item = score_cache.get((baseline_name, bucket, probe, repeat))
                    item = score_cache.get((spec.name, bucket, probe, repeat))
                    if base_item is None or item is None:
                        continue
                    if not np.array_equal(base_item["indices"], item["indices"]):
                        continue
                    stats = bootstrap_delta(
                        item["labels"],
                        base_item["scores"],
                        item["scores"],
                        rng,
                        args.bootstrap_iters,
                    )
                    stats["repeat"] = repeat
                    per_repeat.append(stats)
                finite = np.array(
                    [item["delta_auc_mean"] for item in per_repeat if np.isfinite(item["delta_auc_mean"])],
                    dtype=np.float64,
                )
                results["paired_bootstrap"][spec.name][bucket][probe] = {
                    "baseline": baseline_name,
                    "delta_auc_mean_across_repeats": float(np.mean(finite)) if finite.size else float("nan"),
                    "repeats": per_repeat,
                }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {args.output}")


if __name__ == "__main__":
    main()
