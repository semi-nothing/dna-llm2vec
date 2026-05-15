"""
Base-pair mutagenesis maps for DNA foundation models.

Unlike token-level categorical Jacobians, this script mutates DNA bases in the
raw sequence and re-tokenizes each mutant. That makes the attribution comparable
across BPE, k-mer, byte-level, and single-base tokenizers.

For each mutable base position i and substitution b in A/C/G/T, it computes a
pooled sequence embedding delta:

    delta[i, b] = embed(seq with base i -> b) - embed(original seq)

The default contact-style map is cosine similarity between these mutation-effect
vectors across positions, averaged over substitutions:

    contact[i, j] = mean_b cosine(delta[i, b], delta[j, b])

This is not the same object as gLM2's token-level categorical Jacobian, but it
is the fair version for DNAGPT/DNABERT-style tokenizers.

Examples
--------
  python src/bp_mutagenesis_jacobian_dna.py \
      --loader dnagpt \
      --model "M0:dnagpt/human_gpt2-v1:causal" \
      --sequence ACGTACGTACGTACGT \
      --output-prefix ./figures/bp_jac_m0_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import step4_evaluate as step4  # noqa: E402
from categorical_jacobian_dna import ensure_padding, _load_wrapper  # noqa: E402


DNA_BASES = ("A", "C", "G", "T")
COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _read_fasta(path: str) -> str:
    parts: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts.append(line)
    return "".join(parts)


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def mutable_bp_positions(sequence: str, alphabet: tuple[str, ...]) -> list[int]:
    allowed = set(alphabet) | {x.lower() for x in alphabet}
    return [i for i, ch in enumerate(sequence) if ch in allowed]


def mutate_base(sequence: str, pos: int, base: str) -> str:
    current = sequence[pos]
    replacement = base.lower() if current.islower() else base.upper()
    return sequence[:pos] + replacement + sequence[pos + 1 :]


def tokenize_batch(tokenizer, sequences: list[str], max_length: int, device: str):
    ensure_padding(tokenizer)
    enc = tokenizer(
        sequences,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _evo_backbone_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    hidden = backbone.embedding_layer.embed(input_ids)
    hidden, _ = backbone.stateless_forward(hidden, padding_mask=attention_mask)
    if backbone.norm is not None:
        hidden = backbone.norm(hidden)
    return hidden


def forward_hidden(model, enc: dict[str, torch.Tensor]):
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    evo_hidden = _evo_backbone_hidden(model, input_ids, attention_mask)
    if evo_hidden is not None:
        return evo_hidden

    if hasattr(model, "transformer"):
        out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    else:
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            out = model(**kwargs, use_cache=False, output_hidden_states=False)
        except TypeError:
            out = model(**kwargs)

    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    raise TypeError(f"Unsupported model output type {type(out)!r}")


@torch.no_grad()
def encode_pooled(
    model,
    tokenizer,
    sequences: list[str],
    max_length: int,
    batch_size: int,
    device: str,
    pooling: str,
):
    model.eval()
    pooled_all: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenize_batch(tokenizer, batch, max_length, device)
        hidden = forward_hidden(model, enc)
        pooled = step4.pool_hidden_states(
            hidden=hidden,
            attention_mask=enc["attention_mask"],
            input_ids=enc["input_ids"],
            pooling=pooling,
            eos_token_id=tokenizer.eos_token_id,
        )
        pooled = F.normalize(pooled, dim=-1)
        pooled_all.append(pooled.detach().cpu().float())
        del enc, hidden, pooled
        if device == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(pooled_all, dim=0).numpy()


def build_mutants(sequence: str, positions: list[int], alphabet: tuple[str, ...], include_self: bool):
    mutants: list[str] = []
    metadata: list[tuple[int, str, str]] = []
    for pos in positions:
        original = sequence[pos].upper()
        for base in alphabet:
            if not include_self and base.upper() == original:
                continue
            mutants.append(mutate_base(sequence, pos, base))
            metadata.append((pos, original, base.upper()))
    return mutants, metadata


def delta_to_contact(delta: np.ndarray, metric: str, diag: str, apc: bool):
    """delta shape: [L, A, D], with NaNs for skipped substitutions."""
    valid = np.isfinite(delta).all(axis=2)
    filled = np.where(np.isfinite(delta), delta, 0.0)
    num_pos = delta.shape[0]
    contact = np.zeros((num_pos, num_pos), dtype=np.float32)

    if metric == "effect_norm":
        scores = np.sqrt(np.square(filled).sum(axis=2))
        counts = valid.sum(axis=1).clip(min=1)
        effect = scores.sum(axis=1) / counts
        contact = np.sqrt(effect[:, None] * effect[None, :]).astype(np.float32)
    else:
        for i in range(num_pos):
            for j in range(num_pos):
                vals = []
                for a in range(delta.shape[1]):
                    if not (valid[i, a] and valid[j, a]):
                        continue
                    vi = filled[i, a]
                    vj = filled[j, a]
                    if metric == "dot":
                        vals.append(float(np.dot(vi, vj)))
                    else:
                        denom = np.linalg.norm(vi) * np.linalg.norm(vj)
                        vals.append(float(np.dot(vi, vj) / denom)) if denom > 1e-12 else vals.append(0.0)
                contact[i, j] = float(np.mean(vals)) if vals else 0.0

    if diag == "remove":
        np.fill_diagonal(contact, 0.0)
    if apc:
        total = contact.sum()
        if abs(total) > 1e-12:
            contact = contact - contact.sum(0, keepdims=True) * contact.sum(1, keepdims=True) / total
    if diag == "remove":
        np.fill_diagonal(contact, 0.0)
    return contact


def save_contact_csv(path: str, contact: np.ndarray, positions: list[int], sequence: str):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["i", "j", "i_bp", "j_bp", "i_base", "j_base", "value"])
        for ii, pos_i in enumerate(positions):
            for jj, pos_j in enumerate(positions):
                writer.writerow([
                    ii + 1,
                    jj + 1,
                    pos_i + 1,
                    pos_j + 1,
                    sequence[pos_i],
                    sequence[pos_j],
                    float(contact[ii, jj]),
                ])


def save_contact_png(path: str, contact: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    vmax = np.percentile(contact, 99) if np.isfinite(contact).any() else 1.0
    im = ax.imshow(contact, cmap="Blues", vmin=float(np.min(contact)), vmax=float(vmax))
    ax.set_title(title)
    ax.set_xlabel("mutable bp index")
    ax.set_ylabel("mutable bp index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model spec: name:path:mode")
    p.add_argument(
        "--loader",
        choices=("generic", "dnagpt", "maskedlm", "dnabert2", "caduceus", "evo"),
        default="generic",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence")
    src.add_argument("--fasta")
    p.add_argument("--alphabet", nargs="+", default=list(DNA_BASES))
    p.add_argument("--include-self", action="store_true", help="Also evaluate no-op substitutions")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mutant-batch-size", type=int, default=16)
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--start-bp", type=int, default=1, help="1-based inclusive start position")
    p.add_argument("--end-bp", type=int, default=None, help="1-based inclusive end position")
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--metric", choices=("cosine", "dot", "effect_norm"), default="cosine")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-apc", action="store_true")
    p.add_argument("--output-prefix", required=True)
    p.add_argument(
        "--rc-consistency",
        action="store_true",
        help="Also run reverse complement and save a flipped contact map/correlation.",
    )
    return p.parse_args()


def run_one(args, sequence: str, suffix: str = ""):
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if args.fp32 or device == "cpu" else torch.bfloat16
    spec = step4.ModelSpec.parse(args.model)
    model, tokenizer = _load_wrapper(args.loader)(spec, device, dtype)
    ensure_padding(tokenizer)

    alphabet = tuple(base.upper() for base in args.alphabet)
    positions = mutable_bp_positions(sequence, alphabet)
    start_idx = max(args.start_bp - 1, 0)
    end_idx = len(sequence) if args.end_bp is None else min(args.end_bp, len(sequence))
    positions = [p for p in positions if start_idx <= p < end_idx]
    if args.max_positions is not None:
        positions = positions[: args.max_positions]
    if not positions:
        raise ValueError("No mutable A/C/G/T positions found in the selected region.")

    base_emb = encode_pooled(
        model, tokenizer, [sequence], args.max_length, 1, device, args.pooling
    )[0]
    mutants, metadata = build_mutants(sequence, positions, alphabet, args.include_self)
    mutant_emb = encode_pooled(
        model, tokenizer, mutants, args.max_length, args.mutant_batch_size, device, args.pooling
    )

    delta = np.full((len(positions), len(alphabet), base_emb.shape[0]), np.nan, dtype=np.float32)
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
    base_to_idx = {base: idx for idx, base in enumerate(alphabet)}
    for emb, (pos, _original, new_base) in zip(mutant_emb, metadata):
        delta[pos_to_idx[pos], base_to_idx[new_base]] = emb - base_emb

    contact = delta_to_contact(delta, metric=args.metric, diag="remove", apc=not args.no_apc)
    prefix = args.output_prefix + suffix
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.savez_compressed(
        f"{prefix}.npz",
        delta=delta,
        contact=contact,
        positions=np.array(positions, dtype=np.int32),
        alphabet=np.array(alphabet, dtype=object),
        sequence=np.array(sequence, dtype=object),
    )
    save_contact_csv(f"{prefix}.csv", contact, positions, sequence)
    save_contact_png(f"{prefix}.png", contact, f"{spec.name} bp mutagenesis {args.metric}")
    with open(f"{prefix}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": asdict(spec),
                "loader": args.loader,
                "sequence_length_bp": len(sequence),
                "selected_positions": len(positions),
                "alphabet": alphabet,
                "include_self": args.include_self,
                "pooling": args.pooling,
                "metric": args.metric,
                "max_length": args.max_length,
            },
            fh,
            indent=2,
        )
    print(f"Wrote {prefix}.npz/.csv/.png/.meta.json")
    return contact


def main():
    args = parse_args()
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    sequence = sequence.strip()
    contact = run_one(args, sequence)

    if args.rc_consistency:
        rc_sequence = reverse_complement(sequence)
        rc_contact = run_one(args, rc_sequence, suffix="_rc")
        flipped = rc_contact[::-1, ::-1]
        n = min(contact.shape[0], flipped.shape[0])
        corr = float(np.corrcoef(contact[:n, :n].ravel(), flipped[:n, :n].ravel())[0, 1])
        np.savez_compressed(f"{args.output_prefix}_rc_compare.npz", contact=contact, rc_flipped=flipped, corr=corr)
        print(f"RC contact correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
