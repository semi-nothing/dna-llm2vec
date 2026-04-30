"""
Architecture-specific bidirectional patching utilities for HyenaDNA.

This does NOT remove a Transformer attention mask. Instead, it converts
the causal Hyena sequence-mixing operator into a bidirectional receptive-field
version by switching the FFT-convolution padding scheme, following the
experimental implementation described in the official HyenaDNA repository.
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
    Bidirectional Hyena FFT convolution following the official experimental
    implementation: pad half the sequence before and half after, then crop
    back to the original length after inverse FFT.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size) / fft_size

    padded_length = seqlen + 2 * (seqlen // 2)
    pad_before = padded_length // 2 - (seqlen // 2)
    pad_after = padded_length - seqlen - pad_before
    padded_u = F.pad(u, (pad_before, pad_after), mode="constant", value=0)
    u_f = torch.fft.rfft(padded_u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


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

    for _, module in model.named_modules():
        if not _is_hyena_filter_module(module):
            continue

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
