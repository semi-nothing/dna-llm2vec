"""
Shared helpers for the HyenaDNA experimental branch.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


DEFAULT_HYENA_MODEL = "LongSafari/hyenadna-small-32k-seqlen-hf"


def load_hyena_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hyena_backbone(model_name_or_path: str, device: str = "cpu"):
    """
    Try a few conservative load paths for HyenaDNA.

    We prefer robustness over aggressive optimisation here.
    """
    path = os.path.abspath(model_name_or_path) if os.path.exists(model_name_or_path) else model_name_or_path

    errors: list[str] = []

    try:
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model = model.to(device=device)
        model.eval()
        return model, "AutoModel"
    except Exception as e:
        errors.append(f"AutoModel failed: {e}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model = model.to(device=device)
        model.eval()
        return model, "AutoModelForCausalLM"
    except Exception as e:
        errors.append(f"AutoModelForCausalLM failed: {e}")

    raise RuntimeError(
        "Could not load HyenaDNA with the current environment.\n" + "\n".join(errors)
    )


def count_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
