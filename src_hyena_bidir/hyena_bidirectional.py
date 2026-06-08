"""
Honest bidirectional HyenaDNA patch.

This variant keeps the HyenaDNA causal branch intact, adds an untied reverse
branch, and merges the two directions with a learnable channel gate. It avoids
the old tied 0.5 * (forward + reverse) constraint that forced both directions
to share the same taps.
"""

from __future__ import annotations

import copy
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
    operators_with_reverse_branch: int = 0
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


def _run_hyena_branch(self, projected_u, short_filter, filter_fn, l_filter):
    uc = short_filter(projected_u)[..., :l_filter]
    *x, v = uc.split(self.d_model, dim=1)

    k = filter_fn.filter(l_filter)[0]
    k = k.transpose(0, 1).reshape(self.order - 1, self.d_model, l_filter)
    bias = filter_fn.bias.reshape(self.order - 1, self.d_model)

    for o, x_i in enumerate(reversed(x[1:])):
        v = self.dropout(v * x_i)
        v = filter_fn(v, l_filter, k=k[o], bias=bias[o])

    return v * x[0]


def _hyena_operator_forward_honest_bidirectional(self, u):
    l = u.size(-2)
    l_filter = min(l, self.l_max)
    projected = self.in_proj(u).transpose(1, 2)

    forward = _run_hyena_branch(self, projected, self.short_filter, self.filter_fn, l_filter)
    backward = torch.flip(
        _run_hyena_branch(
            self,
            torch.flip(projected, dims=[-1]),
            self.reverse_short_filter,
            self.reverse_filter_fn,
            l_filter,
        ),
        dims=[-1],
    )

    gate = torch.sigmoid(self.direction_gate).view(1, self.d_model, 1)
    y = gate * forward + (1.0 - gate) * backward
    y = y.transpose(1, 2)
    return self.out_proj(y)


def _attach_reverse_branch(module) -> bool:
    created = False
    if not hasattr(module, "reverse_short_filter"):
        module.reverse_short_filter = copy.deepcopy(module.short_filter)
        created = True
    if not hasattr(module, "reverse_filter_fn"):
        module.reverse_filter_fn = copy.deepcopy(module.filter_fn)
        created = True
    if not hasattr(module, "direction_gate"):
        module.direction_gate = nn.Parameter(torch.zeros(module.d_model))
        created = True
    return created


def make_hyenadna_bidirectional(model):
    """
    Convert a causal HyenaDNA checkpoint into an honest bidirectional variant.

    The reverse branch is initialized as a copy of the pretrained causal branch,
    but it owns separate trainable parameters. The merge gate is initialized at
    0.5 per channel and can learn direction-specific weighting during Stage 2.
    """
    if hasattr(model, "config"):
        model.config.bidirectional = True
        model.config.hyena_bidirectional_patch = True
        model.config.hyena_bidirectional_patch_impl = "honest_untied_two_branch_v1"

    total_operators = 0
    operator_forward_patched = 0

    for _, module in list(model.named_modules()):
        if not _is_hyena_operator_module(module):
            continue
        total_operators += 1
        setattr(module, "bidirectional", True)
        _attach_reverse_branch(module)
        if (
            not getattr(module, "_hyena_honest_bidirectional_operator_forward_patched", False)
            or not _is_forward_bound_to(module, _hyena_operator_forward_honest_bidirectional)
        ):
            module._hyena_operator_original_forward = module.forward
            module.forward = types.MethodType(_hyena_operator_forward_honest_bidirectional, module)
            module._hyena_honest_bidirectional_operator_forward_patched = True
        operator_forward_patched += 1

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
    operators_with_reverse_branch = 0
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
            if all(
                hasattr(module, name)
                for name in ("reverse_short_filter", "reverse_filter_fn", "direction_gate")
            ):
                operators_with_reverse_branch += 1
            if (
                getattr(module, "_hyena_honest_bidirectional_operator_forward_patched", False)
                and _is_forward_bound_to(module, _hyena_operator_forward_honest_bidirectional)
            ):
                operators_forward_patched += 1

    return HyenaBidirectionalReport(
        total_hyena_filters=total_filters,
        modules_with_bidirectional_attr=modules_with_bidir_attr,
        modules_with_bidirectional_true=modules_with_bidir_true,
        modules_forward_patched=operators_forward_patched,
        total_hyena_operators=total_operators,
        operators_with_reverse_branch=operators_with_reverse_branch,
        operators_forward_patched=operators_forward_patched,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
