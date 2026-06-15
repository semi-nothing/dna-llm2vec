"""
Shared helpers for the honest-bidirectional HyenaDNA branch.

Most behavior is inherited from `src_hyena.common`; the local overrides make
saved checkpoints with reverse-branch parameters reload correctly.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from transformers import AutoModel, AutoModelForCausalLM

from hyena_bidirectional import maybe_activate_hyenadna_bidirectional


ROOT = os.path.dirname(os.path.dirname(__file__))
OLD_HYENA_DIR = os.path.join(ROOT, "src_hyena")
_OLD_COMMON_PATH = os.path.join(OLD_HYENA_DIR, "common.py")
_spec = importlib.util.spec_from_file_location("_old_hyena_common", _OLD_COMMON_PATH)
_old_common = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
import sys

sys.modules[_spec.name] = _old_common
_spec.loader.exec_module(_old_common)


DEFAULT_HYENA_MODEL = _old_common.DEFAULT_HYENA_MODEL
resolve_path = _old_common.resolve_path
load_hyena_tokenizer = _old_common.load_hyena_tokenizer
count_parameters_m = _old_common.count_parameters_m
count_trainable_parameters_m = _old_common.count_trainable_parameters_m
save_hyena_checkpoint = _old_common.save_hyena_checkpoint
extract_hidden_states = _old_common.extract_hidden_states
mean_pool_embeddings = _old_common.mean_pool_embeddings
ensure_attention_mask = _old_common.ensure_attention_mask
encode_sequence = _old_common.encode_sequence


def _reload_local_state_after_patch(model, model_name_or_path: str):
    path = resolve_path(model_name_or_path)
    state_path = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else None
    if state_path is None or not os.path.isfile(state_path):
        return model

    state_dict = torch.load(state_path, map_location="cpu")
    model_state = model.state_dict()
    if any(k.startswith("hyena.backbone.") for k in state_dict) and not any(
        k.startswith("hyena.backbone.") for k in model_state
    ):
        state_dict = {
            k.removeprefix("hyena."): v
            for k, v in state_dict.items()
            if k.startswith("hyena.")
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    reverse_missing = [k for k in missing if "reverse_" in k or "direction_gate" in k]
    reverse_unexpected = [k for k in unexpected if "reverse_" in k or "direction_gate" in k]
    print(
        "  Honest-bidir reload      : "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"reverse/gate missing={len(reverse_missing)}, "
        f"reverse/gate unexpected={len(reverse_unexpected)}"
    )
    if reverse_missing:
        print(f"  Honest-bidir missing sample: {reverse_missing[:3]}")
    if reverse_unexpected:
        print(f"  Honest-bidir unexpected sample: {reverse_unexpected[:3]}")
    return model


def _checkpoint_uses_causal_lm_wrapper(model_name_or_path: str) -> bool | None:
    path = resolve_path(model_name_or_path)
    state_path = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else None
    if state_path is None or not os.path.isfile(state_path):
        return None

    state_dict = torch.load(state_path, map_location="cpu")
    if any(k.startswith("hyena.backbone.") for k in state_dict):
        return True
    if any(k.startswith("backbone.") for k in state_dict):
        return False
    return None


def _move_and_prepare(model, model_name_or_path: str, device: str):
    model = maybe_activate_hyenadna_bidirectional(model)
    model = _reload_local_state_after_patch(model, model_name_or_path)
    _old_common._patch_hyena_attention_mask(model)
    model = model.to(device=device)
    model.eval()
    return model


def load_hyena_backbone(model_name_or_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    path = resolve_path(model_name_or_path)
    errors: list[str] = []
    checkpoint_uses_causal_lm = _checkpoint_uses_causal_lm_wrapper(model_name_or_path)
    loader_order = (
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModel", AutoModel),
    ) if checkpoint_uses_causal_lm else (
        ("AutoModel", AutoModel),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    )

    for loader_name, loader in loader_order:
        try:
            model = loader.from_pretrained(
                path,
                trust_remote_code=True,
                dtype=dtype,
            )
            return _move_and_prepare(model, model_name_or_path, device), loader_name
        except Exception as e:
            errors.append(f"{loader_name} failed: {e}")

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
    model = _move_and_prepare(model, model_name_or_path, device)
    return model, "AutoModelForCausalLM"
