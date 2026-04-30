"""
Step 1 for the HyenaDNA branch: environment + loader sanity check.

Unlike the DNAGPT branch, there is no causal-attention mask to patch here.
This script simply verifies that a HyenaDNA checkpoint can be loaded in the
current environment and that it exposes a usable hidden-state path.
"""

import argparse

import torch

from common import DEFAULT_HYENA_MODEL, count_parameters_m, load_hyena_backbone, load_hyena_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="HyenaDNA Step 1: loader sanity check")
    p.add_argument("--model", default=DEFAULT_HYENA_MODEL)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("HyenaDNA  |  Step 1: Loader Sanity Check")
    print("=" * 64)
    print(f"  Model  : {args.model}")
    print(f"  Device : {device}")

    tokenizer = load_hyena_tokenizer(args.model)
    model, load_path = load_hyena_backbone(args.model, device=device)

    print(f"  Load path    : {load_path}")
    print(f"  Parameters   : {count_parameters_m(model):.1f}M")
    print(f"  Tokenizer    : vocab={len(tokenizer):,}")
    print(f"  pad/eos ids  : {tokenizer.pad_token_id} / {tokenizer.eos_token_id}")

    sample = "ACGT" * (args.max_length // 4)
    enc = tokenizer(
        [sample],
        truncation=True,
        max_length=args.max_length,
        padding="longest",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        hidden = out.last_hidden_state
        output_kind = "last_hidden_state"
    elif hasattr(out, "hidden_states") and out.hidden_states is not None:
        hidden = out.hidden_states[-1]
        output_kind = "hidden_states[-1]"
    elif isinstance(out, tuple):
        hidden = out[0]
        output_kind = "tuple[0]"
    else:
        raise RuntimeError(f"Unsupported HyenaDNA output type: {type(out)!r}")

    print(f"  Output kind  : {output_kind}")
    print(f"  Hidden shape : {tuple(hidden.shape)}")
    print("  Sanity check : PASSED")


if __name__ == "__main__":
    main()
