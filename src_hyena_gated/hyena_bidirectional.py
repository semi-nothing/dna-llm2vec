"""
Gated tied bidirectional HyenaDNA patch.

This is the conservative ablation of the original `src_hyena` bidirectional
logic: forward and reverse directions still share the same Hyena taps, but the
fixed 0.5 average is replaced by learnable per-channel gates.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HyenaBidirectionalReport:
    total_hyena_filters: int
    modules_with_bidirectional_attr: int
    modules_with_bidirectional_true: int
    modules_forward_patched: int
    total_hyena_operators: int = 0
    operators_with_direction_gate: int = 0
    operators_forward_patched: int = 0
    trainable_params: int = 0
    total_params: int = 0


def _fftconv_causal(u: torch.Tensor, k: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    work_dtype = torch.float32
    u_work = u.to(work_dtype)
    k_work = k.to(work_dtype)
    d_work = d.to(work_dtype)

    k_f = torch.fft.rfft(k_work, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u_work, n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u_work * d_work.unsqueeze(-1)).to(dtype=u.dtype)


def _is_hyena_filter_module(module) -> bool:
    type_name = type(module).__name__.lower()
    return (
        "hyenafilter" in type_name
        or (
            hasattr(module, "filter")
            and hasattr(module, "bias")
            and hasattr(module, "implicit_filter")
            and hasattr(module, "pos_emb")
        )
    )


def _is_hyena_operator_module(module) -> bool:
    type_name = type(module).__name__.lower()
    return (
        "hyenaoperator" in type_name
        or (
            hasattr(module, "short_filter")
            and hasattr(module, "filter_fn")
            and hasattr(module, "in_proj")
            and hasattr(module, "out_proj")
            and hasattr(module, "order")
            and hasattr(module, "d_model")
        )
    )


def _is_forward_bound_to(module, fn) -> bool:
    forward = getattr(module, "forward", None)
    return (
        getattr(forward, "__self__", None) is module
        and getattr(forward, "__func__", None) is fn
    )


def _hyena_filter_forward_gated_bidirectional(self, x, L, k=None, bias=None, *args, **kwargs):
    if k is None:
        k = self.filter(L)
    k = k[0] if type(k) is tuple else k
    if bias is None:
        bias = self.bias
    if getattr(self, "fused_fft_conv", False):
        raise NotImplementedError(
            "Gated bidirectional Hyena patch currently supports only the unfused FFT path."
        )

    forward = _fftconv_causal(x, k, bias)
    backward = torch.flip(
        _fftconv_causal(torch.flip(x, dims=[-1]), k, bias),
        dims=[-1],
    )
    gate = torch.sigmoid(self.direction_gate[: forward.size(1)]).view(1, -1, 1)
    return gate * forward + (1.0 - gate) * backward


def _short_filter_gated_bidirectional(self, u: torch.Tensor, target_len: int) -> torch.Tensor:
    forward = self.short_filter(u)[..., :target_len]
    backward = torch.flip(
        self.short_filter(torch.flip(u, dims=[-1]))[..., :target_len],
        dims=[-1],
    )
    gate = torch.sigmoid(self.short_direction_gate).view(1, -1, 1)
    return gate * forward + (1.0 - gate) * backward


def _hyena_operator_forward_gated_bidirectional(self, u):
    l = u.size(-2)
    l_filter = min(l, self.l_max)
    u = self.in_proj(u).transpose(1, 2)

    uc = _short_filter_gated_bidirectional(self, u, l_filter)
    *x, v = uc.split(self.d_model, dim=1)

    k = self.filter_fn.filter(l_filter)[0]
    k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
    bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

    for o, x_i in enumerate(reversed(x[1:])):
        v = self.dropout(v * x_i)
        v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

    y = (v * x[0]).transpose(1, 2)
    return self.out_proj(y)


def _attach_filter_gate(module) -> None:
    if not hasattr(module, "direction_gate"):
        module.direction_gate = nn.Parameter(torch.zeros(module.d_model))


def _attach_operator_gate(module) -> None:
    if not hasattr(module, "short_direction_gate"):
        module.short_direction_gate = nn.Parameter(torch.zeros(module.short_filter.out_channels))


def make_hyenadna_bidirectional(model):
    """
    Convert a causal HyenaDNA checkpoint into a tied bidirectional variant with
    learnable non-symmetric direction gates.

    Gates are initialized at zero logits, so sigmoid(g)=0.5 and the initial
    behavior matches the old fixed average. Stage 2 can then move each channel
    toward forward- or reverse-dominant use.
    """
    if hasattr(model, "config"):
        model.config.bidirectional = True
        model.config.hyena_bidirectional_patch = True
        model.config.hyena_bidirectional_patch_impl = "tied_learned_gate_v1"

    total_filters = 0
    filter_forward_patched = 0
    total_operators = 0
    operator_forward_patched = 0

    fused_filters = []

    for name, module in model.named_modules():
        if _is_hyena_filter_module(module):
            if getattr(module, "fused_fft_conv", False):
                fused_filters.append(name)
                continue
            total_filters += 1
            setattr(module, "bidirectional", True)
            _attach_filter_gate(module)
            if (
                not getattr(module, "_hyena_gated_bidirectional_forward_patched", False)
                or not _is_forward_bound_to(module, _hyena_filter_forward_gated_bidirectional)
            ):
                module._hyena_original_forward = module.forward
                module.forward = types.MethodType(_hyena_filter_forward_gated_bidirectional, module)
                module._hyena_gated_bidirectional_forward_patched = True
            filter_forward_patched += 1
            continue

        if _is_hyena_operator_module(module):
            total_operators += 1
            setattr(module, "bidirectional", True)
            _attach_operator_gate(module)
            if (
                not getattr(module, "_hyena_gated_bidirectional_operator_forward_patched", False)
                or not _is_forward_bound_to(module, _hyena_operator_forward_gated_bidirectional)
            ):
                module._hyena_operator_original_forward = module.forward
                module.forward = types.MethodType(_hyena_operator_forward_gated_bidirectional, module)
                module._hyena_gated_bidirectional_operator_forward_patched = True
            operator_forward_patched += 1

    if fused_filters:
        raise NotImplementedError(
            "Gated bidirectional Hyena patch supports only the unfused FFT path; "
            f"found fused_fft_conv=True in {len(fused_filters)} HyenaFilter module(s), "
            f"including {fused_filters[:3]}."
        )
    if total_filters == 0:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not find any HyenaFilter-like modules to patch."
        )
    if total_operators == 0:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not find any HyenaOperator-like modules to patch."
        )

    return model


def maybe_activate_hyenadna_bidirectional(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    wants_bidir = bool(
        getattr(cfg, "hyena_bidirectional_patch", False)
        or getattr(cfg, "bidirectional", False)
    )
    if wants_bidir:
        return make_hyenadna_bidirectional(model)
    return model


def inspect_hyenadna_bidirectional(model) -> HyenaBidirectionalReport:
    total_filters = 0
    modules_with_bidir_attr = 0
    modules_with_bidir_true = 0
    filter_forward_patched = 0
    total_operators = 0
    operators_with_direction_gate = 0
    operators_forward_patched = 0

    for _, module in model.named_modules():
        if _is_hyena_filter_module(module):
            total_filters += 1
            if hasattr(module, "bidirectional"):
                modules_with_bidir_attr += 1
            if getattr(module, "bidirectional", False):
                modules_with_bidir_true += 1
            if (
                getattr(module, "_hyena_gated_bidirectional_forward_patched", False)
                and _is_forward_bound_to(module, _hyena_filter_forward_gated_bidirectional)
            ):
                filter_forward_patched += 1

        if _is_hyena_operator_module(module):
            total_operators += 1
            if hasattr(module, "short_direction_gate"):
                operators_with_direction_gate += 1
            if (
                getattr(module, "_hyena_gated_bidirectional_operator_forward_patched", False)
                and _is_forward_bound_to(module, _hyena_operator_forward_gated_bidirectional)
            ):
                operators_forward_patched += 1

    return HyenaBidirectionalReport(
        total_hyena_filters=total_filters,
        modules_with_bidirectional_attr=modules_with_bidir_attr,
        modules_with_bidirectional_true=modules_with_bidir_true,
        modules_forward_patched=filter_forward_patched,
        total_hyena_operators=total_operators,
        operators_with_direction_gate=operators_with_direction_gate,
        operators_forward_patched=operators_forward_patched,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
