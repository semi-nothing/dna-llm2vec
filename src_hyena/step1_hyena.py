"""
Step 1 for the HyenaDNA branch.

H0: pretrained causal HyenaDNA baseline
H1: pretrained weights + bidirectional Hyena receptive-field patch

This is NOT attention-mask removal. The patch converts the causal Hyena
sequence-mixing operator into a bidirectional receptive-field version by
switching the FFT-convolution padding scheme.
"""

from __future__ import annotations

import argparse

import torch

from common import (
    DEFAULT_HYENA_MODEL,
    count_parameters_m,
    count_trainable_parameters_m,
    extract_hidden_states,
    load_hyena_causal_lm,
    load_hyena_tokenizer,
    save_hyena_checkpoint,
)
from hyena_bidirectional import inspect_hyenadna_bidirectional, make_hyenadna_bidirectional


def parse_args():
    p = argparse.ArgumentParser(description="HyenaDNA Step 1: bidirectional receptive-field patch")
    p.add_argument("--model", default=DEFAULT_HYENA_MODEL)
    p.add_argument("--output", default="./hyena_bidir_h1")
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def _print_report(report):
    print(f"  HyenaFilter modules        : {report.total_hyena_filters}")
    print(f"  Modules with bidir attr    : {report.modules_with_bidirectional_attr}")
    print(f"  Modules with bidir=True    : {report.modules_with_bidirectional_true}")
    print(f"  Modules forward-patched    : {report.modules_forward_patched}")
    print(f"  Trainable params           : {report.trainable_params / 1e6:.2f}M")
    print(f"  Total params               : {report.total_params / 1e6:.2f}M")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("HyenaDNA  |  Step 1: Bidirectional Receptive-Field Patch")
    print("=" * 72)
    print(f"  H0 model : {args.model}")
    print(f"  H1 save  : {args.output}")
    print(f"  Device   : {device}")

    tokenizer = load_hyena_tokenizer(args.model)
    model, load_path = load_hyena_causal_lm(args.model, device=device)

    print(f"  Load path                : {load_path}")
    print(f"  Parameters               : {count_parameters_m(model):.2f}M")
    print(f"  Trainable params         : {count_trainable_parameters_m(model):.2f}M")
    print(f"  Tokenizer vocab          : {len(tokenizer):,}")
    print(f"  pad/eos ids              : {tokenizer.pad_token_id} / {tokenizer.eos_token_id}")

    model = make_hyenadna_bidirectional(model)
    report = inspect_hyenadna_bidirectional(model)
    _print_report(report)

    sample = "ACGT" * max(1, args.max_length // 4)
    enc = tokenizer(
        [sample],
        truncation=True,
        max_length=args.max_length,
        padding="longest",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.inference_mode():
        out = model(
            input_ids=enc["input_ids"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = extract_hidden_states(out)

    print(f"  Forward output kind      : hidden_states")
    print(f"  Hidden shape             : {tuple(hidden.shape)}")
    print("  Step 1 sanity            : PASSED")

    save_hyena_checkpoint(model, tokenizer, args.output)
    print(f"  Saved H1 checkpoint      : {args.output}")


if __name__ == "__main__":
    main()
