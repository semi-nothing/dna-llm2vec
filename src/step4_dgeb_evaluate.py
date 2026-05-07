#!/usr/bin/env python
"""
Evaluate frozen DNA encoders on DGEB DNA tasks.

This script adapts the repository's existing Step 4 embedding path to the DGEB
BioSeqTransformer interface. It does not train a downstream model itself; DGEB
chooses the official evaluator for each task.

Example:
  uv run python src/step4_dgeb_evaluate.py \
    --models \
      "M0:dnagpt/human_gpt2-v1:causal" \
      "M5:./contrastive_dnagpt_crop_lora_fn_mask_only_s42_r1:bidir" \
      "H5:./hyena_h5_crop_contrastive_ep1_s42_r1:hyena" \
    --task-set recommended \
    --batch-size 128 \
    --max-length 1024 \
    --hyena-max-length 8192 \
    --output ./eval_results/dgeb_dna
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
HYENA_DIR = os.path.join(ROOT, "src_hyena")
if HYENA_DIR not in sys.path:
    sys.path.insert(0, HYENA_DIR)

import step4_evaluate as base  # noqa: E402
from common import count_parameters_m, extract_hidden_states, load_hyena_backbone, load_hyena_tokenizer  # noqa: E402


RECOMMENDED_DGEB_DNA_TASKS = [
    "bac_16S_phylogeny",
    "arch_16S_phylogeny",
    "euk_18S_phylogeny",
    "rpob_bac_dna_phylogeny",
    "rpob_arch_dna_phylogeny",
    "ecoli_rna_clustering",
]


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
        return ModelSpec(name=parts[0], path=parts[1], mode=parts[2])


def _infer_embed_dim(model) -> int:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model", "embed_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None:
        for attr in ("embedding_dim", "out_features"):
            value = getattr(embeddings, attr, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer embedding dimension for DGEB model metadata.")


def encode_hyena_sequences(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str = "mean",
    desc: str = "Encoding",
) -> np.ndarray:
    if pooling != "mean":
        raise ValueError("Hyena DGEB evaluation currently supports mean pooling only.")

    from tqdm.auto import tqdm
    import torch.nn.functional as F
    from common import ensure_attention_mask

    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc=desc, leave=False):
        batch = sequences[i : i + batch_size]
        with torch.inference_mode():
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding="longest",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            attention_mask = ensure_attention_mask(enc, tokenizer.pad_token_id)

            encoder = model.hyena if hasattr(model, "hyena") else model
            out = encoder(
                input_ids=enc["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = extract_hidden_states(out)
            pooled = base.pool_hidden_states(
                hidden=hidden,
                attention_mask=attention_mask,
                input_ids=enc["input_ids"],
                pooling=pooling,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            pooled = F.normalize(pooled, dim=-1)

        embeddings.append(pooled.cpu().float().numpy())
        del enc, out, hidden, pooled

    return np.concatenate(embeddings, axis=0)


class FrozenDNAEncoder:
    """
    Minimal DGEB-compatible BioSeqTransformer wrapper.

    DGEB expects encode() to return shape [N, num_layers, D]. Our Step 4 path
    emits one pooled final-layer representation, so num_layers is reported as 1.
    """

    def __init__(
        self,
        name: str,
        model,
        tokenizer,
        encode_fn: Callable,
        batch_size: int,
        max_length: int,
        device: str,
        pooling: str,
    ):
        self.id = name
        self.hf_name = name
        self.encoder = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_length
        self.device = torch.device(device)
        self.pooling = pooling
        self.layers = [0]
        self.layer_labels = ["last"]
        self.num_param = sum(p.numel() for p in model.parameters())
        self._embed_dim = _infer_embed_dim(model)
        self._encode_fn = encode_fn

    @property
    def metadata(self) -> dict:
        return {
            "hf_name": self.hf_name,
            "num_layers": self.num_layers,
            "num_params": self.num_param,
            "embed_dim": self.embed_dim,
        }

    @property
    def modality(self):
        from dgeb.modality import Modality

        return Modality.DNA

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode(self, sequences, **kwargs) -> np.ndarray:
        if not isinstance(sequences, list):
            sequences = list(sequences)
        emb = self._encode_fn(
            model=self.encoder,
            tokenizer=self.tokenizer,
            sequences=sequences,
            batch_size=self.batch_size,
            max_length=self.max_seq_length,
            device=str(self.device),
            pooling=self.pooling,
            desc=f"DGEB {self.hf_name}",
        )
        return emb[:, None, :]


def load_dgeb_model(spec: ModelSpec, args, device: str, dtype: torch.dtype) -> FrozenDNAEncoder:
    if spec.mode == "hyena":
        if args.pooling != "mean":
            raise ValueError(
                "Hyena DGEB evaluation currently supports only --pooling mean. "
                f"Got --pooling {args.pooling!r} for model {spec.name!r}."
            )
        path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
        print(f"  Loading {spec.name} ({path}, mode=hyena)")
        tokenizer = load_hyena_tokenizer(path)
        model, load_path = load_hyena_backbone(path, device=device, dtype=dtype)
        print(f"    Load path  : {load_path}")
        print(f"    Parameters : {count_parameters_m(model):.1f}M | vocab: {len(tokenizer):,}")
        return FrozenDNAEncoder(
            name=spec.name,
            model=model,
            tokenizer=tokenizer,
            encode_fn=encode_hyena_sequences,
            batch_size=args.hyena_batch_size or args.batch_size,
            max_length=args.hyena_max_length,
            device=device,
            pooling=args.pooling,
        )

    generic = base.ModelSpec(name=spec.name, path=spec.path, mode=spec.mode)
    model, tokenizer = base.load_model(generic, device=device, dtype=dtype)
    return FrozenDNAEncoder(
        name=spec.name,
        model=model,
        tokenizer=tokenizer,
        encode_fn=base.encode_sequences,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
    )


def resolve_tasks(args):
    try:
        import dgeb
        import dgeb.tasks  # noqa: F401 - registers Task subclasses
        from dgeb.modality import Modality
    except ImportError as e:
        raise SystemExit(
            "DGEB is not installed. Install it with `uv pip install dgeb` "
            "or add `dgeb>=0.2.0` to the environment."
        ) from e

    if args.tasks:
        return dgeb.get_tasks_by_name(args.tasks)
    if args.task_set == "recommended":
        return dgeb.get_tasks_by_name(RECOMMENDED_DGEB_DNA_TASKS)
    if args.task_set == "all-dna":
        return dgeb.get_tasks_by_modality(Modality.DNA)
    raise ValueError(f"Unknown task set {args.task_set!r}")


def summarize_results(results) -> dict:
    summary = {}
    for result in results:
        task_id = result.task.id
        layer = result.results[0]
        metrics = {metric.id: metric.value for metric in layer.metrics}
        summary[task_id] = {
            "primary_metric": result.task.primary_metric_id,
            "primary_value": metrics.get(result.task.primary_metric_id),
            "metrics": metrics,
        }
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Run DGEB DNA embedding evaluation.")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Model specs: "NAME:PATH:MODE"; MODE is causal, bidir, encoder, or hyena.',
    )
    task_group = p.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task-set",
        choices=("recommended", "all-dna"),
        default="recommended",
        help="DGEB DNA task subset to run when --tasks is not provided.",
    )
    task_group.add_argument("--tasks", nargs="+", default=None, help="Explicit DGEB task ids to run.")
    p.add_argument("--output", default="./eval_results/dgeb_dna", help="Output directory.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hyena-batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024, help="Max tokens for DNAGPT/generic models.")
    p.add_argument("--hyena-max-length", type=int, default=8192, help="Max tokens/bp for HyenaDNA models.")
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fp32", action="store_true", help="Use float32 instead of bf16/fp16 on CUDA.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    tasks = resolve_tasks(args)
    import dgeb

    specs = [ModelSpec.parse(s) for s in args.models]
    if args.pooling != "mean" and any(spec.mode == "hyena" for spec in specs):
        raise SystemExit(
            "Hyena DGEB evaluation currently supports only --pooling mean. "
            "Use --pooling mean or run Hyena and DNAGPT models separately."
        )
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.fp32 or device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("DGEB DNA evaluation")
    print(f"  Device      : {device}")
    print(f"  Dtype       : {dtype}")
    print(f"  Task set    : {args.task_set}")
    print(f"  Tasks       : {[task.metadata.id for task in tasks]}")
    print(f"  Output      : {args.output}")

    all_summaries = {}
    for spec in specs:
        print(f"\n-> {spec.name}")
        model = load_dgeb_model(spec, args, device=device, dtype=dtype)
        evaluation = dgeb.DGEB(tasks=tasks, seed=args.seed)
        results = evaluation.run(model, output_folder=args.output)
        all_summaries[spec.name] = summarize_results(results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_set": args.task_set,
                "tasks": [task.metadata.id for task in tasks],
                "models": [spec.__dict__ for spec in specs],
                "results": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
