"""
Shared helpers for the HyenaDNA experimental branch.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from hyena_bidirectional import maybe_activate_hyenadna_bidirectional


DEFAULT_HYENA_MODEL = "LongSafari/hyenadna-small-32k-seqlen-hf"


def resolve_path(model_name_or_path: str) -> str:
    return os.path.abspath(model_name_or_path) if os.path.exists(model_name_or_path) else model_name_or_path


def load_hyena_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_path(model_name_or_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _move_and_prepare(model, device: str):
    model = maybe_activate_hyenadna_bidirectional(model)
    model = model.to(device=device)
    model.eval()
    return model


def load_hyena_backbone(model_name_or_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """
    Conservative loader for HyenaDNA backbone models.

    The local H1/H2 checkpoints may have been saved from a CausalLM class,
    so we try AutoModel first and then fall back to AutoModelForCausalLM.
    """
    path = resolve_path(model_name_or_path)
    errors: list[str] = []

    try:
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            dtype=dtype,
        )
        return _move_and_prepare(model, device), "AutoModel"
    except Exception as e:
        errors.append(f"AutoModel failed: {e}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            dtype=dtype,
        )
        return _move_and_prepare(model, device), "AutoModelForCausalLM"
    except Exception as e:
        errors.append(f"AutoModelForCausalLM failed: {e}")

    raise RuntimeError(
        "Could not load HyenaDNA with the current environment.\n" + "\n".join(errors)
    )


def load_hyena_causal_lm(model_name_or_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    path = resolve_path(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = maybe_activate_hyenadna_bidirectional(model)
    model = model.to(device=device)
    return model, "AutoModelForCausalLM"


def count_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def save_hyena_checkpoint(model, tokenizer, output_dir: str):
    """
    Save a HyenaDNA checkpoint without going through transformers'
    tied-weight cleanup path, which currently trips over the shared
    Sin.freq parameters in the remote-code implementation.
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "config") and model.config is not None:
        model.config.save_pretrained(output_dir)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def extract_hidden_states(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states[-1]
    if isinstance(output, tuple):
        if len(output) >= 2 and isinstance(output[1], (tuple, list)) and output[1]:
            return output[1][-1]
        return output[0]
    raise TypeError(
        f"Unsupported HyenaDNA output type {type(output)!r}; "
        "expected last_hidden_state, hidden_states, or tuple output."
    )


def mean_pool_embeddings(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def ensure_attention_mask(enc: dict[str, torch.Tensor], pad_token_id: int | None) -> torch.Tensor:
    if "attention_mask" in enc:
        return enc["attention_mask"]
    input_ids = enc["input_ids"]
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return (input_ids != pad_token_id).long()


def encode_sequence(
    model,
    tokenizer,
    sequence: str,
    pooling: str = "mean",
    max_length: int = 1024,
    device: str | None = None,
) -> torch.Tensor:
    if pooling != "mean":
        raise ValueError(f"Unsupported pooling={pooling!r}; HyenaDNA default is mean pooling.")

    if device is None:
        device = next(model.parameters()).device.type

    enc = tokenizer(
        [sequence],
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    attention_mask = ensure_attention_mask(enc, tokenizer.pad_token_id)

    with torch.inference_mode():
        model_inputs = {"input_ids": enc["input_ids"], "output_hidden_states": True, "return_dict": True}
        out = model(**model_inputs)
        hidden = extract_hidden_states(out)
        pooled = mean_pool_embeddings(hidden, attention_mask)
        pooled = F.normalize(pooled, dim=-1)

    return pooled.squeeze(0).detach().cpu()
