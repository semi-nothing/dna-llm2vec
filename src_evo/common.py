"""
Shared helpers for the Evo LoRA experimental branch.

Evo is loaded through Hugging Face remote code. The helper functions here keep
the rest of the pipeline close to the DNAGPT/HyenaDNA scripts while isolating
the Evo-specific hidden-state extraction path.
"""

from __future__ import annotations

import inspect
import os
import shutil
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_EVO_MODEL = "togethercomputer/evo-1-131k-base"
DEFAULT_EVO_REVISION = os.environ.get("EVO_REVISION", "1.1_fix")
VALID_BASES_RE = __import__("re").compile(r"[^ACGT]")


def resolve_path(model_name_or_path: str) -> str:
    return os.path.abspath(model_name_or_path) if os.path.exists(model_name_or_path) else model_name_or_path


def set_existing_byte_pad_token(tokenizer) -> None:
    """Set padding to an existing token without resizing Evo embeddings."""
    if tokenizer.pad_token is not None and tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is not None:
            return
    for candidate in (" ", "\n", "\t", "\x00", "\xff"):
        unk_id = getattr(tokenizer, "unk_token_id", None)
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if token_id is not None and token_id != unk_id:
            tokenizer.pad_token = candidate
            if tokenizer.pad_token_id is not None:
                return
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and (unk_id is None or ids[0] != unk_id):
            tokenizer.pad_token = tokenizer.decode(ids)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = int(ids[0])
            return
    raise ValueError("Could not choose an existing Evo tokenizer token for padding.")


def load_evo_tokenizer(model_name_or_path: str):
    path = resolve_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        revision=DEFAULT_EVO_REVISION,
    )
    set_existing_byte_pad_token(tokenizer)
    return tokenizer


def load_evo_causal_lm(
    model_name_or_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    path = resolve_path(model_name_or_path)
    config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        revision=DEFAULT_EVO_REVISION,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        trust_remote_code=True,
        revision=DEFAULT_EVO_REVISION,
        torch_dtype=dtype,
    )
    model = maybe_activate_evo_bidirectional(model)
    model = model.to(device=device)
    return model, "AutoModelForCausalLM"


def _copy_if_present(src_path: str | None, output_dir: str) -> None:
    if not src_path or not os.path.isfile(src_path):
        return
    dst_path = os.path.join(output_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        shutil.copy2(src_path, dst_path)


def copy_remote_code_artifacts(model, tokenizer, output_dir: str) -> None:
    candidates: list[str | None] = []
    for obj in (
        model,
        getattr(model, "backbone", None),
        tokenizer,
        getattr(model, "config", None),
    ):
        if obj is None:
            continue
        try:
            candidates.append(inspect.getfile(obj.__class__))
        except (TypeError, OSError):
            pass
    for path in candidates:
        _copy_if_present(path, output_dir)


def save_evo_checkpoint(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    copy_remote_code_artifacts(model, tokenizer, output_dir)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(output_dir, safe_serialization=False)
        except TypeError:
            model.save_pretrained(output_dir)
    else:
        if getattr(model, "config", None) is not None:
            model.config.save_pretrained(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def peft_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def ensure_mask_token(tokenizer, model=None) -> str:
    if getattr(tokenizer, "mask_token", None) is not None:
        return tokenizer.mask_token
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if model is not None and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.mask_token


def ensure_attention_mask(enc: dict[str, torch.Tensor], pad_token_id: int | None) -> torch.Tensor:
    if "attention_mask" in enc:
        return enc["attention_mask"]
    input_ids = enc["input_ids"]
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return (input_ids != pad_token_id).long()


def prepare_evo_batch(tokenizer, batch: list[str], max_length: int, device: str):
    """Use Evo's prepare_batch helper when installed; otherwise use tokenizer."""
    try:
        from evo.scoring import prepare_batch
    except ImportError:
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = ensure_attention_mask(enc, tokenizer.pad_token_id)
        return enc["input_ids"], attention_mask

    input_ids, seq_lengths = prepare_batch(
        batch,
        tokenizer,
        prepend_bos=False,
        device=device,
    )
    if not torch.is_tensor(seq_lengths):
        seq_lengths = torch.as_tensor(seq_lengths, device=device)
    else:
        seq_lengths = seq_lengths.to(device=device)
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]
        seq_lengths = seq_lengths.clamp(max=max_length)
    offsets = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    attention_mask = (offsets < seq_lengths.unsqueeze(1)).long()
    return input_ids, attention_mask


def extract_hidden_states(out: Any) -> torch.Tensor:
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, tuple):
        if len(out) >= 2 and isinstance(out[-1], (tuple, list)):
            return out[-1][-1]
        if torch.is_tensor(out[0]) and out[0].dim() == 3:
            return out[0]
    raise TypeError(f"Unsupported Evo output type {type(out)!r}.")


def evo_backbone_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    model = peft_base_model(model)
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    if not all(hasattr(backbone, name) for name in ("embedding_layer", "stateless_forward", "norm")):
        return None
    if not hasattr(backbone.embedding_layer, "embed"):
        return None
    hidden = backbone.embedding_layer.embed(input_ids)
    try:
        hidden, _ = backbone.stateless_forward(hidden, padding_mask=attention_mask)
    except TypeError:
        hidden, _ = backbone.stateless_forward(hidden)
    if backbone.norm is not None:
        hidden = backbone.norm(hidden)
    return hidden


def evo_hidden_states(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    try:
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        return extract_hidden_states(out)
    except TypeError as e:
        msg = str(e)
        if "output_hidden_states" not in msg and "use_cache" not in msg:
            raise
    hidden = evo_backbone_hidden(model, input_ids, attention_mask)
    if hidden is None:
        raise RuntimeError("Could not extract Evo hidden states from public forward or backbone internals.")
    return hidden


def mean_pool_embeddings(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def count_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def infer_hidden_dim(model) -> int:
    model = peft_base_model(model)
    cfg = getattr(model, "config", None)
    for attr in ("n_embd", "hidden_size", "d_model", "dim", "hidden_dim"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[1])
    raise RuntimeError("Could not infer Evo hidden dimension.")


def infer_lora_target_modules(model, requested: str) -> list[str]:
    if requested and requested.lower() != "auto":
        return [x.strip() for x in requested.split(",") if x.strip()]

    preferred_tokens = ("proj", "linear", "dense", "wq", "wk", "wv", "wo", "qkv", "mlp")
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower = name.lower()
        if any(skip in lower for skip in ("lm_head", "embed", "embedding")):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if any(tok in lower for tok in preferred_tokens):
            suffixes.add(leaf)
    if not suffixes:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                lower = name.lower()
                if not any(skip in lower for skip in ("lm_head", "embed", "embedding")):
                    suffixes.add(name.rsplit(".", 1)[-1])
    if not suffixes:
        raise RuntimeError("Could not infer Evo LoRA target modules; pass --lora-target-modules explicitly.")
    return sorted(suffixes)


def make_evo_bidirectional(model):
    """
    Best-effort Evo E1 patch.

    Evo remote code has changed over revisions. We persist an explicit config
    marker and set common causal/bidirectional flags on StripedHyena-like
    modules. If a local revision exposes more specific controls, this marker
    lets downstream loaders reapply the same best-effort activation.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.evo_bidirectional_patch = True
        cfg.bidirectional = True
        cfg.use_cache = False

    patched = 0
    for _, module in model.named_modules():
        changed = False
        for attr, value in (
            ("causal", False),
            ("is_causal", False),
            ("use_causal_conv", False),
            ("bidirectional", True),
        ):
            if hasattr(module, attr):
                try:
                    setattr(module, attr, value)
                    changed = True
                except Exception:
                    pass
        if changed:
            patched += 1
    model._evo_bidirectional_modules_patched = patched
    return model


def maybe_activate_evo_bidirectional(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    if bool(getattr(cfg, "evo_bidirectional_patch", False) or getattr(cfg, "bidirectional", False)):
        return make_evo_bidirectional(model)
    return model
