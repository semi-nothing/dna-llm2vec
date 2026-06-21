"""
Shared helpers for the gated-tied HyenaDNA branch.
"""

from __future__ import annotations

import importlib.util
import os
import re

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
    projection_missing = [k for k in missing if "direction_projection" in k]
    projection_unexpected = [k for k in unexpected if "direction_projection" in k]
    print(
        "  Projected-bidir reload   : "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"projection missing={len(projection_missing)}, "
        f"projection unexpected={len(projection_unexpected)}"
    )
    if projection_missing:
        print(f"  Projected-bidir missing sample: {projection_missing[:3]}")
    if projection_unexpected:
        print(f"  Projected-bidir unexpected sample: {projection_unexpected[:3]}")

    # Fail loud if any *trained* weight failed to load. The freshly attached
    # direction projections are allowed to be missing (first patch falls back to
    # the 0.5/0.5 average init); so are the model's documented ignore-on-load
    # buffers (e.g. Sin.freq). Anything else missing means the checkpoint
    # silently did not restore real weights -> abort instead of degrading.
    ignore_patterns = list(
        getattr(getattr(model, "config", None), "_keys_to_ignore_on_load_missing", None) or []
    )

    def _is_allowed_missing(key: str) -> bool:
        if "direction_projection" in key:
            return True
        return any(re.search(pattern, key) for pattern in ignore_patterns)

    critical_missing = [k for k in missing if not _is_allowed_missing(k)]
    if critical_missing:
        raise RuntimeError(
            "Projected-bidir reload failed to restore trained weights "
            f"({len(critical_missing)} keys), e.g. {critical_missing[:8]}. "
            "Aborting instead of running with partially-initialized weights."
        )
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


def load_hyena_backbone(model_name_or_path: str, device: str = "cpu", dtype: torch.dtype | None = torch.float32):
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
            load_kwargs = {"trust_remote_code": True}
            if dtype is not None:
                load_kwargs["dtype"] = dtype
            model = loader.from_pretrained(path, **load_kwargs)
            return _move_and_prepare(model, model_name_or_path, device), loader_name
        except Exception as e:
            errors.append(f"{loader_name} failed: {e}")

    raise RuntimeError(
        "Could not load HyenaDNA with the current environment.\n" + "\n".join(errors)
    )


def load_hyena_causal_lm(model_name_or_path: str, device: str = "cpu", dtype: torch.dtype | None = torch.float32):
    path = resolve_path(model_name_or_path)
    load_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model = _move_and_prepare(model, model_name_or_path, device)
    return model, "AutoModelForCausalLM"
