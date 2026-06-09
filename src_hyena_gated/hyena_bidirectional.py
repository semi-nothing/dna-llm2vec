"""
Tied bidirectional HyenaDNA patch with concat projection.

This ablation keeps the same Hyena taps for both directions:
  forward  = original causal Hyena branch
  backward = original causal Hyena branch on the reversed sequence, flipped back

The only new trainable parameters are per-layer 2d -> d direction projections
that mix concat([forward, backward]) back into the residual stream width. The
projection is initialized to the old fixed average, but can learn asymmetric
direction mixing during Stage 2/3 full fine-tuning.
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
    operators_with_direction_projection: int = 0
    operators_forward_patched: int = 0
    trainable_params: int = 0
    total_params: int = 0


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


def _run_tied_hyena_branch(self, projected_u: torch.Tensor, l_filter: int) -> torch.Tensor:
    uc = self.short_filter(projected_u)[..., :l_filter]
    *x, v = uc.split(self.d_model, dim=1)

    k = self.filter_fn.filter(l_filter)[0]
    k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
    bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

    for o, x_i in enumerate(reversed(x[1:])):
        v = self.dropout(v * x_i)
        v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

    return v * x[0]


def _hyena_operator_forward_projected_bidirectional(self, u):
    l = u.size(-2)
    l_filter = min(l, self.l_max)
    projected = self.in_proj(u).transpose(1, 2)

    forward = _run_tied_hyena_branch(self, projected, l_filter)
    backward = torch.flip(
        _run_tied_hyena_branch(self, torch.flip(projected, dims=[-1]), l_filter),
        dims=[-1],
    )

    y = torch.cat([forward, backward], dim=1).transpose(1, 2)
    y = self.direction_projection(y)
    return self.out_proj(y)


def _attach_direction_projection(module) -> None:
    if hasattr(module, "direction_projection"):
        return

    ref = module.out_proj.weight
    projection = nn.Linear(2 * module.d_model, module.d_model, bias=True)
    projection = projection.to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        eye = torch.eye(module.d_model, device=ref.device, dtype=ref.dtype)
        projection.weight[:, : module.d_model].copy_(0.5 * eye)
        projection.weight[:, module.d_model :].copy_(0.5 * eye)
    module.direction_projection = projection


def make_hyenadna_bidirectional(model):
    """
    Convert a causal HyenaDNA checkpoint into a tied bidirectional variant with
    a learnable concat projection.
    """
    if hasattr(model, "config"):
        model.config.bidirectional = True
        model.config.hyena_bidirectional_patch = True
        model.config.hyena_bidirectional_patch_impl = "tied_concat_projection_v1"

    total_operators = 0
    operator_forward_patched = 0

    for _, module in model.named_modules():
        if not _is_hyena_operator_module(module):
            continue
        total_operators += 1
        setattr(module, "bidirectional", True)
        _attach_direction_projection(module)
        if (
            not getattr(module, "_hyena_projected_bidirectional_operator_forward_patched", False)
            or not _is_forward_bound_to(module, _hyena_operator_forward_projected_bidirectional)
        ):
            module._hyena_operator_original_forward = module.forward
            module.forward = types.MethodType(_hyena_operator_forward_projected_bidirectional, module)
            module._hyena_projected_bidirectional_operator_forward_patched = True
        operator_forward_patched += 1

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
    total_operators = 0
    operators_with_projection = 0
    operators_forward_patched = 0

    for _, module in model.named_modules():
        if _is_hyena_filter_module(module):
            total_filters += 1
            if hasattr(module, "bidirectional"):
                modules_with_bidir_attr += 1
            if getattr(module, "bidirectional", False):
                modules_with_bidir_true += 1

        if _is_hyena_operator_module(module):
            total_operators += 1
            if hasattr(module, "direction_projection"):
                operators_with_projection += 1
            if (
                getattr(module, "_hyena_projected_bidirectional_operator_forward_patched", False)
                and _is_forward_bound_to(module, _hyena_operator_forward_projected_bidirectional)
            ):
                operators_forward_patched += 1

    return HyenaBidirectionalReport(
        total_hyena_filters=total_filters,
        modules_with_bidirectional_attr=modules_with_bidir_attr,
        modules_with_bidirectional_true=modules_with_bidir_true,
        modules_forward_patched=operators_forward_patched,
        total_hyena_operators=total_operators,
        operators_with_direction_projection=operators_with_projection,
        operators_forward_patched=operators_forward_patched,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
