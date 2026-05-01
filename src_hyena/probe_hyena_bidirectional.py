"""
Probe whether the HyenaDNA bidirectional patch introduces right-context
influence into left-side hidden states.

The core test compares two inputs that differ only in a suffix region:
  x      = original sequence
  x_tail = same prefix, mutated suffix

For a truly bidirectional model, hidden states in the untouched left prefix
should change when the right suffix changes. For a causal model, they should
remain nearly unchanged (up to numerical noise).

By default:
  - H0 is the pretrained causal HyenaDNA checkpoint
  - H1 is created in-memory by applying make_hyenadna_bidirectional(H0)

Optionally, you can provide --h1-model to probe a saved H1/H2 checkpoint
instead of patching H0 on the fly.
"""

from __future__ import annotations

import argparse
import copy

import torch

from common import DEFAULT_HYENA_MODEL, extract_hidden_states, load_hyena_causal_lm, load_hyena_tokenizer
from hyena_bidirectional import inspect_hyenadna_bidirectional, make_hyenadna_bidirectional


_MUTATION_TABLE = str.maketrans({
    "A": "C",
    "C": "G",
    "G": "T",
    "T": "A",
    "N": "N",
})


def parse_args():
    p = argparse.ArgumentParser(description="Probe HyenaDNA bidirectional patch via suffix-perturbation hidden-state tests")
    p.add_argument("--model", default=DEFAULT_HYENA_MODEL, help="Base causal HyenaDNA checkpoint / model id (H0)")
    p.add_argument("--h1-model", default=None, help="Optional saved H1/H2 checkpoint to probe instead of patching H0 in-memory")
    p.add_argument("--max-length", type=int, default=256, help="Token length passed to the tokenizer")
    p.add_argument("--suffix-fraction", type=float, default=0.25, help="Fraction of the sequence mutated at the right end")
    p.add_argument("--left-window-fraction", type=float, default=0.25, help="Fraction of the sequence used as the left-side sensitivity window")
    p.add_argument("--right-window-fraction", type=float, default=0.25, help="Fraction of the sequence used as the right-side sensitivity window")
    p.add_argument("--pooling", choices=("mean", "max"), default="mean", help="How to aggregate hidden-state deltas inside each window")
    p.add_argument("--layerwise", action="store_true", help="Also print per-layer suffix-perturbation sensitivity")
    return p.parse_args()


def build_sequence(length: int) -> str:
    motif = "ACGT"
    return (motif * ((length + len(motif) - 1) // len(motif)))[:length]


def mutate_suffix(seq: str, suffix_len: int) -> str:
    suffix_len = max(1, min(len(seq), suffix_len))
    cut = len(seq) - suffix_len
    return seq[:cut] + seq[cut:].translate(_MUTATION_TABLE)


def encode_pair(tokenizer, seq_a: str, seq_b: str, max_length: int, device: str):
    batch = tokenizer(
        [seq_a, seq_b],
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )
    if "attention_mask" not in batch:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long)
        else:
            batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    return {k: v.to(device) for k, v in batch.items()}


def forward_hidden(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    return extract_hidden_states(out)


def forward_hidden_stack(model, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    with torch.inference_mode():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(out, "hidden_states", None)
    if not hidden_states:
        return [extract_hidden_states(out)]
    return list(hidden_states)


def aggregate_delta(delta: torch.Tensor, start: int, end: int, pooling: str) -> float:
    window = delta[:, start:end]
    if window.numel() == 0:
        return float("nan")
    if pooling == "max":
        return float(window.max().item())
    return float(window.mean().item())


def describe_probe(name: str, hidden: torch.Tensor, valid_len: int, left_len: int, right_len: int, pooling: str) -> dict[str, float]:
    delta = (hidden[0] - hidden[1]).abs().mean(dim=-1)  # (T,)
    left = aggregate_delta(delta.unsqueeze(0), 0, left_len, pooling)
    right = aggregate_delta(delta.unsqueeze(0), valid_len - right_len, valid_len, pooling)
    global_mean = float(delta[:valid_len].mean().item())
    return {
        "model": name,
        "left_window_delta": left,
        "right_window_delta": right,
        "global_delta": global_mean,
    }


def print_summary(summary: dict[str, float]):
    print(
        f"  {summary['model']:<3} "
        f"left={summary['left_window_delta']:.8f}  "
        f"right={summary['right_window_delta']:.8f}  "
        f"global={summary['global_delta']:.8f}"
    )


def print_layerwise_probe(
    name: str,
    hidden_stack: list[torch.Tensor],
    valid_len: int,
    left_len: int,
    right_len: int,
    pooling: str,
):
    print(f"\n{name} layer-wise sensitivity")
    print("  layer  left_delta    right_delta   global_delta")
    for layer_idx, hidden in enumerate(hidden_stack):
        summary = describe_probe(f"{name}-L{layer_idx}", hidden, valid_len, left_len, right_len, pooling)
        print(
            f"  {layer_idx:>3d}   "
            f"{summary['left_window_delta']:.8f}  "
            f"{summary['right_window_delta']:.8f}  "
            f"{summary['global_delta']:.8f}"
        )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 72)
    print("HyenaDNA  |  Bidirectional Patch Probe")
    print("=" * 72)
    print(f"  H0 model                : {args.model}")
    print(f"  H1 source               : {args.h1_model or 'in-memory patch of H0'}")
    print(f"  Device                  : {device}")
    print(f"  Precision               : {dtype}")

    tokenizer = load_hyena_tokenizer(args.model)
    h0_model, load_path = load_hyena_causal_lm(args.model, device=device, dtype=dtype)
    print(f"  H0 load path            : {load_path}")

    if args.h1_model:
        h1_model, h1_load_path = load_hyena_causal_lm(args.h1_model, device=device, dtype=dtype)
        print(f"  H1 load path            : {h1_load_path}")
    else:
        h1_model = copy.deepcopy(h0_model)
        h1_model = make_hyenadna_bidirectional(h1_model)
        print("  H1 patch                : make_hyenadna_bidirectional(H0)")

    report = inspect_hyenadna_bidirectional(h1_model)
    print(f"  H1 HyenaFilter modules  : {report.total_hyena_filters}")
    print(f"  H1 bidir=True modules   : {report.modules_with_bidirectional_true}")
    print(f"  H1 patched forwards     : {report.modules_forward_patched}")

    seq_len = max(16, args.max_length - tokenizer.num_special_tokens_to_add(pair=False))
    suffix_len = max(1, int(round(seq_len * args.suffix_fraction)))
    left_len = max(1, int(round(seq_len * args.left_window_fraction)))
    right_len = max(1, int(round(seq_len * args.right_window_fraction)))

    seq = build_sequence(seq_len)
    seq_mut = mutate_suffix(seq, suffix_len)
    batch = encode_pair(tokenizer, seq, seq_mut, args.max_length, device)
    valid_len = int(batch["attention_mask"][0].sum().item())

    print(f"  Sequence length         : {seq_len}")
    print(f"  Tokenized valid length  : {valid_len}")
    print(f"  Mutated suffix length   : {suffix_len}")
    print(f"  Left window length      : {left_len}")
    print(f"  Right window length     : {right_len}")
    print(f"  Delta pooling           : {args.pooling}")

    h0_hidden = forward_hidden(h0_model, batch)
    h1_hidden = forward_hidden(h1_model, batch)

    h0_summary = describe_probe("H0", h0_hidden, valid_len, left_len, right_len, args.pooling)
    h1_summary = describe_probe("H1", h1_hidden, valid_len, left_len, right_len, args.pooling)

    print("\nSuffix perturbation sensitivity")
    print("  Hidden-state delta between original input and suffix-mutated input")
    print_summary(h0_summary)
    print_summary(h1_summary)

    left_ratio = h1_summary["left_window_delta"] / max(h0_summary["left_window_delta"], 1e-12)
    global_ratio = h1_summary["global_delta"] / max(h0_summary["global_delta"], 1e-12)
    print(f"\n  H1/H0 left-delta ratio  : {left_ratio:.4f}")
    print(f"  H1/H0 global-delta ratio: {global_ratio:.4f}")

    if h1_summary["left_window_delta"] > (h0_summary["left_window_delta"] * 5.0):
        verdict = "Patch introduces substantial right-context influence into left-prefix states."
    elif h1_summary["left_window_delta"] > (h0_summary["left_window_delta"] * 1.5):
        verdict = "Patch shows moderate right-context influence; residual causal paths may still matter."
    else:
        verdict = "Patch shows weak left-prefix sensitivity; short-filter causality may still dominate."
    print(f"  Verdict                 : {verdict}")

    if args.layerwise:
        h0_stack = forward_hidden_stack(h0_model, batch)
        h1_stack = forward_hidden_stack(h1_model, batch)
        print_layerwise_probe("H0", h0_stack, valid_len, left_len, right_len, args.pooling)
        print_layerwise_probe("H1", h1_stack, valid_len, left_len, right_len, args.pooling)


if __name__ == "__main__":
    main()
