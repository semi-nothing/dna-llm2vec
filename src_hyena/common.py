"""
Shared helpers for the HyenaDNA experimental branch.
"""

from __future__ import annotations

import inspect
import os
import shutil
import types
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput

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


def _mask_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return hidden_states * mask


def _configure_pad_embedding(model) -> None:
    pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_token_id is None:
        return

    if not hasattr(model, "get_input_embeddings"):
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return
    if pad_token_id < 0 or pad_token_id >= embeddings.weight.shape[0]:
        return

    embeddings.padding_idx = pad_token_id
    with torch.no_grad():
        embeddings.weight[pad_token_id].zero_()


def _copy_if_present(src_path: str | None, output_dir: str) -> None:
    if not src_path:
        return
    if not os.path.isfile(src_path):
        return
    dst_path = os.path.join(output_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return
    shutil.copy2(src_path, dst_path)


def _copy_hyena_remote_code_artifacts(model, tokenizer, output_dir: str) -> None:
    candidates: list[str | None] = []

    for obj in (
        model,
        getattr(model, "hyena", None),
        getattr(getattr(model, "hyena", None), "backbone", None),
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

    for filename in ("configuration_hyena.py", "modeling_hyena.py", "tokenization_hyena.py"):
        existing = os.path.join(output_dir, filename)
        if os.path.isfile(existing):
            continue
        for path in candidates:
            if path and os.path.basename(path) == filename:
                _copy_if_present(path, output_dir)
                break


def _patch_hyena_attention_mask(model) -> None:
    if getattr(model, "_hyena_attention_mask_patched", False):
        return

    backbone = None
    core_model = None
    if hasattr(model, "hyena") and hasattr(model.hyena, "backbone"):
        core_model = model.hyena
        backbone = model.hyena.backbone
    elif hasattr(model, "backbone"):
        core_model = model
        backbone = model.backbone

    if backbone is None or core_model is None:
        return

    original_backbone_forward = backbone.forward

    def backbone_forward_with_attention_mask(
        self,
        input_ids,
        inputs_embeds=None,
        output_hidden_states=False,
        attention_mask=None,
    ):
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        hidden_states = _mask_hidden_states(hidden_states, attention_mask)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(layer.__call__, hidden_states)
            else:
                hidden_states = layer(hidden_states)
            hidden_states = _mask_hidden_states(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states.to(dtype=self.ln_f.weight.dtype))
        hidden_states = _mask_hidden_states(hidden_states, attention_mask)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states

    backbone._hyena_original_forward = original_backbone_forward
    backbone.forward = types.MethodType(backbone_forward_with_attention_mask, backbone)

    original_core_forward = core_model.forward

    def core_forward_with_attention_mask(
        self,
        input_ids,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
        attention_mask=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        if output_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states

    core_model._hyena_original_forward = original_core_forward
    core_model.forward = types.MethodType(core_forward_with_attention_mask, core_model)

    if hasattr(model, "lm_head") and hasattr(model, "hyena"):
        original_lm_forward = model.forward

        def causal_lm_forward_with_attention_mask(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.LongTensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
        ):
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.hyena(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                attention_mask=attention_mask,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states).float()

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.vocab_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
            )

        model._hyena_original_forward = original_lm_forward
        model.forward = types.MethodType(causal_lm_forward_with_attention_mask, model)

    _configure_pad_embedding(model)
    model._hyena_attention_mask_patched = True


def _move_and_prepare(model, device: str):
    model = maybe_activate_hyenadna_bidirectional(model)
    _patch_hyena_attention_mask(model)
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
    _patch_hyena_attention_mask(model)
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
    _copy_hyena_remote_code_artifacts(model, tokenizer, output_dir)

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
        model_inputs["attention_mask"] = attention_mask
        out = model(**model_inputs)
        hidden = extract_hidden_states(out)
        pooled = mean_pool_embeddings(hidden, attention_mask)
        pooled = F.normalize(pooled, dim=-1)

    return pooled.squeeze(0).detach().cpu()
