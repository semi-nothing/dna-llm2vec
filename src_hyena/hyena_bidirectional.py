"""
Architecture-specific bidirectional patching utilities for HyenaDNA.

This does NOT remove a Transformer attention mask. Instead, it converts
the causal Hyena sequence-mixing operator into a bidirectional receptive-field
version by patching both:
  1. the implicit long filter FFT convolution, and
  2. the local short depthwise convolution path,
following the experimental implementation described in the official
HyenaDNA repository.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HyenaBidirectionalReport:
    total_hyena_filters: int
    modules_with_bidirectional_attr: int
    modules_with_bidirectional_true: int
    modules_forward_patched: int
    trainable_params: int
    total_params: int


def hyena_fftconv_bidirectional(u: torch.Tensor, k: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Hyena FFT convolution built from two tied directional paths:
      1. the original causal convolution, and
      2. the same convolution applied to the reversed sequence and flipped back.

    Averaging the two preserves the learned causal filter while making the
    receptive field explicitly bidirectional.
    """
    def _fftconv_causal(u_causal: torch.Tensor, k_causal: torch.Tensor, d_causal: torch.Tensor) -> torch.Tensor:
        seqlen = u_causal.shape[-1]
        fft_size = 2 * seqlen

        work_dtype = torch.float32
        u_work = u_causal.to(work_dtype)
        k_work = k_causal.to(work_dtype)
        d_work = d_causal.to(work_dtype)

        k_f = torch.fft.rfft(k_work, n=fft_size) / fft_size
        u_f = torch.fft.rfft(u_work, n=fft_size)

        if len(u_causal.shape) > 3:
            k_f = k_f.unsqueeze(1)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
        out = y + u_work * d_work.unsqueeze(-1)
        return out.to(dtype=u_causal.dtype)

    forward = _fftconv_causal(u, k, D)
    backward = torch.flip(
        _fftconv_causal(torch.flip(u, dims=[-1]), k, D),
        dims=[-1],
    )
    return 0.5 * (forward + backward)


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


def _hyena_filter_forward_bidirectional(self, x, L, k=None, bias=None, *args, **kwargs):
    if k is None:
        k = self.filter(L)
    k = k[0] if type(k) is tuple else k

    if bias is None:
        bias = self.bias

    if getattr(self, "fused_fft_conv", False):
        raise NotImplementedError(
            "Bidirectional Hyena patch currently supports only the unfused FFT path."
        )

    return hyena_fftconv_bidirectional(x, k, bias)


def _short_filter_bidirectional(conv, u: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Apply tied forward/backward local depthwise convolutions and average them,
    yielding an explicitly bidirectional local receptive field.
    """
    forward = conv(u)[..., :target_len]
    backward = torch.flip(
        conv(torch.flip(u, dims=[-1]))[..., :target_len],
        dims=[-1],
    )
    return 0.5 * (forward + backward)


def _hyena_operator_forward_bidirectional(self, u):
    l = u.size(-2)
    l_filter = min(l, self.l_max)
    u = self.in_proj(u).transpose(1, 2)

    uc = _short_filter_bidirectional(self.short_filter, u, l_filter)
    *x, v = uc.split(self.d_model, dim=1)

    k = self.filter_fn.filter(l_filter)[0]
    k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
    bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

    for o, x_i in enumerate(reversed(x[1:])):
        v = self.dropout(v * x_i)
        v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

    y = (v * x[0]).transpose(1, 2)
    y = self.out_proj(y)
    return y


def make_hyenadna_bidirectional(model):
    """
    Convert a pretrained causal HyenaDNA checkpoint into an H1-style
    bidirectional receptive-field variant without reinitialising weights.
    """
    if hasattr(model, "config"):
        model.config.bidirectional = True
        model.config.hyena_bidirectional_patch = True
        model.config.hyena_bidirectional_patch_impl = "fft_padding_v1"

    total_filters = 0
    modules_with_bidir_attr = 0
    modules_with_bidir_true = 0
    modules_forward_patched = 0
    total_operators = 0
    operator_forward_patched = 0

    for _, module in model.named_modules():
        if _is_hyena_filter_module(module):
            total_filters += 1

            if hasattr(module, "bidirectional"):
                modules_with_bidir_attr += 1
            setattr(module, "bidirectional", True)
            modules_with_bidir_true += 1

            if not getattr(module, "_hyena_bidirectional_forward_patched", False):
                module._hyena_original_forward = module.forward
                module.forward = types.MethodType(_hyena_filter_forward_bidirectional, module)
                module._hyena_bidirectional_forward_patched = True
            modules_forward_patched += 1
            continue

        if _is_hyena_operator_module(module):
            total_operators += 1
            if not getattr(module, "_hyena_bidirectional_operator_forward_patched", False):
                module._hyena_operator_original_forward = module.forward
                module.forward = types.MethodType(_hyena_operator_forward_bidirectional, module)
                module._hyena_bidirectional_operator_forward_patched = True
            operator_forward_patched += 1

    if total_filters == 0:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not find any HyenaFilter-like modules to patch."
        )
    if modules_with_bidir_true != total_filters or modules_forward_patched != total_filters:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not fully patch all HyenaFilter modules.\n"
            f"total_filters={total_filters}, bidirectional_true={modules_with_bidir_true}, "
            f"forward_patched={modules_forward_patched}"
        )
    if total_operators == 0:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not find any HyenaOperator-like modules to patch."
        )
    if operator_forward_patched != total_operators:
        raise RuntimeError(
            "make_hyenadna_bidirectional() did not fully patch all HyenaOperator modules.\n"
            f"total_operators={total_operators}, forward_patched={operator_forward_patched}"
        )

    return model


def maybe_activate_hyenadna_bidirectional(model):
    """
    Re-apply the non-serialised Hyena bidirectional forward patch after loading
    a saved H1/H2 checkpoint from disk.
    """
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
    modules_forward_patched = 0

    for _, module in model.named_modules():
        if not _is_hyena_filter_module(module):
            continue
        total_filters += 1
        if hasattr(module, "bidirectional"):
            modules_with_bidir_attr += 1
        if getattr(module, "bidirectional", False):
            modules_with_bidir_true += 1
        if getattr(module, "_hyena_bidirectional_forward_patched", False):
            modules_forward_patched += 1

    return HyenaBidirectionalReport(
        total_hyena_filters=total_filters,
        modules_with_bidirectional_attr=modules_with_bidir_attr,
        modules_with_bidirectional_true=modules_with_bidir_true,
        modules_forward_patched=modules_forward_patched,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
