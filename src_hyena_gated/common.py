"""
Shared helpers for the gated-tied HyenaDNA branch.
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
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    gate_missing = [k for k in missing if "direction_gate" in k]
    gate_unexpected = [k for k in unexpected if "direction_gate" in k]
    print(
        "  Gated-bidir reload       : "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"gate missing={len(gate_missing)}, gate unexpected={len(gate_unexpected)}"
    )
    if gate_missing:
        print(f"  Gated-bidir missing sample: {gate_missing[:3]}")
    if gate_unexpected:
        print(f"  Gated-bidir unexpected sample: {gate_unexpected[:3]}")
    return model


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

    try:
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            dtype=dtype,
        )
        return _move_and_prepare(model, model_name_or_path, device), "AutoModel"
    except Exception as e:
        errors.append(f"AutoModel failed: {e}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            dtype=dtype,
        )
        return _move_and_prepare(model, model_name_or_path, device), "AutoModelForCausalLM"
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
    model = _move_and_prepare(model, model_name_or_path, device)
    return model, "AutoModelForCausalLM"
