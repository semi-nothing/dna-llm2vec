"""
Evo Step 1: create an E1 bidirectional-patched checkpoint.

This is the Evo counterpart of DNAGPT M1 / HyenaDNA H1. The patch is
best-effort because Evo remote-code internals vary by revision; the saved
checkpoint carries ``config.evo_bidirectional_patch=True`` so downstream Evo
scripts reapply the activation when loading.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    DEFAULT_EVO_MODEL,
    DEFAULT_EVO_REVISION,
    count_parameters_m,
    load_evo_causal_lm,
    load_evo_tokenizer,
    make_evo_bidirectional,
    save_evo_checkpoint,
    verify_evo_bidirectional,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evo Step 1: bidirectional patch")
    p.add_argument("--model", default=DEFAULT_EVO_MODEL)
    p.add_argument("--output", default="./evo_e1_bidir")
    p.add_argument("--verify-max-length", type=int, default=128)
    p.add_argument("--verify-atol", type=float, default=1e-6)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    return p.parse_args()


def _dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    dtype = _dtype(args.dtype)
    print("[Evo Step 1] bidirectional patch")
    print(f"  Model    : {args.model}")
    print(f"  Revision : {DEFAULT_EVO_REVISION}")
    print(f"  Output   : {args.output}")
    tokenizer = load_evo_tokenizer(args.model)
    model, load_path = load_evo_causal_lm(args.model, device=args.device, dtype=dtype)
    model = make_evo_bidirectional(model)
    patched = getattr(model, "_evo_bidirectional_modules_patched", 0)
    print(f"  Load path: {load_path}")
    print(f"  Params   : {count_parameters_m(model):.1f}M")
    print(f"  Patched  : {patched} modules with causal/bidirectional flags")
    if not args.skip_verify:
        report = verify_evo_bidirectional(
            model,
            tokenizer,
            max_length=args.verify_max_length,
            device=args.device,
            atol=args.verify_atol,
        )
        print(
            "  Verify   : "
            f"passed={report['passed']}  "
            f"prefix max diff={report['max_abs_prefix_diff']:.3e}  "
            f"mean diff={report['mean_abs_prefix_diff']:.3e}  "
            f"tokens={report['probe_tokens']}/{report['common_prefix_tokens']}/{report['valid_tokens']}"
        )
        if not report["passed"]:
            raise RuntimeError(
                "Evo bidirectional verification failed: changing future tokens "
                "did not change prefix hidden states. The patch appears causal "
                "or no-op; refusing to save E1."
            )
    save_evo_checkpoint(model, tokenizer, args.output)
    print(f"  Saved    : {args.output}")


if __name__ == "__main__":
    main()
