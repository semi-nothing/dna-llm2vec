"""
HyenaDNA  |  Step 5: Full Fine-tuning Evaluation
================================================

Hyena-specific full fine-tuning entrypoint.

This script reuses the benchmark loading, task loop, metrics, and result
aggregation from ``src/step5_full_finetune_eval.py`` while replacing only the
model-loading and classification-head pieces that are specific to HyenaDNA.

Model spec format:

    name:path:mode

where ``mode`` is normally ``encoder`` for HyenaDNA checkpoints. ``causal`` and
``bidir`` are accepted as aliases for convenience; the actual causal/bidirectional
behaviour is determined by the checkpoint/config and the Hyena bidirectional
activation helpers.

Example:

    uv run python src_hyena/step5_hyena_full_finetune_eval.py \\
      --models \\
        "H0:LongSafari/hyenadna-small-32k-seqlen-hf:encoder" \\
        "H1:./hyena_bidir_h1:encoder" \\
        "H5:./hyena_h5_crop_contrastive_ep1_s42_r1:encoder" \\
      --gue-plus-only \\
      --gue-plus-dir ./data/GUE_plus \\
      --epi-crop-mode junction \\
      --epi-crop-bp 8192 \\
      --max-length 8192 \\
      --epochs 20 \\
      --batch-size 4 \\
      --lr 3e-5 \\
      --warmup-steps 50 \\
      --monitor mcc \\
      --grad-ckpt \\
      --no-wandb \\
      --output ./eval_results/step5full_hyena_epi_s42.json
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
HYENA_SRC = os.path.join(ROOT, "src_hyena")
sys.path.insert(0, SRC)
sys.path.insert(0, HYENA_SRC)

import step5_full_finetune_eval as base  # noqa: E402
from common import (  # noqa: E402
    count_parameters_m,
    extract_hidden_states,
    load_hyena_backbone,
    load_hyena_tokenizer,
)
from hyena_bidirectional import maybe_activate_hyenadna_bidirectional  # noqa: E402
from hyena_bidirectional import inspect_hyenadna_bidirectional  # noqa: E402


@dataclass
class HyenaModelSpec:
    name: str
    path: str
    mode: str

    @staticmethod
    def parse(spec: str) -> "HyenaModelSpec":
        parts = spec.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model spec {spec!r}. Expected name:path:mode, "
                "for example H5:./hyena_h5_crop_contrastive_ep1_s42_r1:encoder"
            )
        name, path, mode = parts
        if mode not in ("encoder", "causal", "bidir"):
            raise ValueError(
                f"Hyena mode must be 'encoder', 'causal', or 'bidir', got {mode!r}."
            )
        return HyenaModelSpec(name=name, path=path, mode=mode)


class HyenaSeqDataset(base.Dataset):
    """Tokenise sequences and synthesize attention masks if the tokenizer omits them."""

    def __init__(self, sequences, labels, tokenizer, max_length: int):
        enc = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        self.input_ids = enc["input_ids"]
        if "attention_mask" in enc:
            self.attention_mask = enc["attention_mask"]
        else:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                self.attention_mask = [[1] * len(x) for x in self.input_ids]
            else:
                self.attention_mask = [[0 if tok == pad_id else 1 for tok in x] for x in self.input_ids]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def pool_hyena_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    if pooling == "average":
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "weighted":
        positions = torch.arange(
            1, hidden.size(1) + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        weights = mask * positions
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if pooling == "last":
        lengths = attention_mask.sum(dim=1).clamp_min(1).long() - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]
    raise ValueError(f"Unsupported pooling={pooling!r}")


def build_classification_head(
    hidden_dim: int,
    n_classes: int,
    head_type: str,
    head_dropout: float,
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(hidden_dim, n_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    raise ValueError(f"Unsupported head_type={head_type!r}")


class HyenaDNAClassifier(nn.Module):
    """Pool final HyenaDNA hidden states, then apply a task head."""

    def __init__(
        self,
        backbone,
        n_classes: int,
        hidden_dim: int,
        pooling: str = "average",
        head_type: str = "linear",
        head_dropout: float = 0.1,
    ):
        super().__init__()
        # The shared training loop deep-copies the loaded base model per task.
        # Re-activate the non-serialized Hyena bidirectional forward patch on
        # that copy to keep H1/H2/H3... behaviour consistent after deepcopy.
        self.backbone = maybe_activate_hyenadna_bidirectional(backbone)
        self.pooling = pooling
        self.classifier = build_classification_head(hidden_dim, n_classes, head_type, head_dropout)

    def forward(self, input_ids, attention_mask):
        # If the checkpoint loaded as AutoModelForCausalLM, use the Hyena core
        # directly. Classification only needs final hidden states, not LM logits
        # over the vocabulary, which are expensive for long 8192-bp inputs.
        core = getattr(self.backbone, "hyena", self.backbone)
        out = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)
        pooled = pool_hyena_hidden(hidden, attention_mask, self.pooling)
        return self.classifier(pooled)


def _infer_hidden_dim(model) -> int:
    config = getattr(model, "config", None)
    for attr in ("n_embd", "hidden_size", "d_model", "dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return int(embeddings.weight.shape[1])

    for attr_path in (
        ("hyena", "backbone", "embeddings"),
        ("backbone", "embeddings"),
    ):
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            return int(obj.weight.shape[1])

    raise RuntimeError("Could not infer HyenaDNA hidden dimension from config or embeddings.")


def _parameter_dtype_summary(model) -> str:
    counts: dict[str, int] = {}
    for param in model.parameters():
        dtype_name = str(param.dtype).replace("torch.", "")
        counts[dtype_name] = counts.get(dtype_name, 0) + param.numel()
    return ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(counts.items()))


def _attach_gradient_checkpointing_enable(model) -> None:
    """Expose a HF-like gradient_checkpointing_enable method for the shared trainer."""

    if hasattr(model, "gradient_checkpointing_enable"):
        return

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            import torch.utils.checkpoint as checkpoint
        except Exception:
            checkpoint = None

        for obj in (
            self,
            getattr(self, "hyena", None),
            getattr(getattr(self, "hyena", None), "backbone", None),
            getattr(self, "backbone", None),
        ):
            if obj is None:
                continue
            if hasattr(obj, "gradient_checkpointing"):
                obj.gradient_checkpointing = True
            if checkpoint is not None and not hasattr(obj, "_gradient_checkpointing_func"):
                use_reentrant = False
                if isinstance(gradient_checkpointing_kwargs, dict):
                    use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", False)

                def _ckpt_func(fn, *args, _use_reentrant=use_reentrant, **kwargs):
                    try:
                        return checkpoint.checkpoint(fn, *args, use_reentrant=_use_reentrant, **kwargs)
                    except TypeError:
                        if kwargs:
                            raise
                        return checkpoint.checkpoint(fn, *args)

                obj._gradient_checkpointing_func = _ckpt_func

    # Attach at class level so per-task copy.deepcopy(base_model) instances also
    # expose the method used by the shared Step 5 training loop.
    setattr(model.__class__, "gradient_checkpointing_enable", gradient_checkpointing_enable)


def load_base_model(spec: HyenaModelSpec, device: str, dtype):
    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, hyena-backbone)")

    tokenizer = load_hyena_tokenizer(path)
    model, load_path = load_hyena_backbone(path, device="cpu", dtype=dtype)
    _attach_gradient_checkpointing_enable(model)
    bidir_report = inspect_hyenadna_bidirectional(model)

    hidden_dim = _infer_hidden_dim(model)
    # The shared Step 5 trainer expects config.n_embd when constructing the head.
    if not hasattr(model, "config") or model.config is None:
        raise RuntimeError("HyenaDNA model has no config object.")
    model.config.n_embd = hidden_dim

    model.eval()
    print(f"    Load path  : {load_path}")
    print(f"    Parameters : {count_parameters_m(model):.1f}M  |  hidden: {hidden_dim}  |  vocab: {len(tokenizer):,}")
    print(f"    Dtypes     : {_parameter_dtype_summary(model)}")
    print(
        "    Bidir patch: "
        f"{bidir_report.modules_forward_patched}/{bidir_report.total_hyena_filters} filters patched"
    )
    return model, tokenizer


# Monkeypatch the generic Step 5 script with Hyena-specific components.
base.ModelSpec = HyenaModelSpec
base.SeqDataset = HyenaSeqDataset
base.DNAClassifier = HyenaDNAClassifier
base.load_base_model = load_base_model


if __name__ == "__main__":
    base.main()
