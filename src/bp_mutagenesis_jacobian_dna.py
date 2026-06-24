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
import gc
import inspect
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

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
    n_records = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                n_records += 1
                continue
            parts.append(line)
    if n_records > 1:
        print(f"WARNING: {path} contains {n_records} FASTA records; concatenating them.")
    return "".join(parts)


def _crop_sequence(seq: str, max_bp: int, mode: str, anchor: int | None = None) -> tuple[str, int]:
    if len(seq) <= max_bp:
        return seq, 0
    if mode == "junction":
        if anchor is None:
            anchor = len(seq) // 2
        start = anchor - (max_bp // 2)
        start = max(0, min(start, len(seq) - max_bp))
    elif mode == "center":
        start = (len(seq) - max_bp) // 2
    else:
        raise ValueError(f"Unknown crop mode {mode!r}")
    return seq[start : start + max_bp], start


def _read_epi_csv(path: str, label: int | None, index: int, crop_bp: int, crop_mode: str):
    matched = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_label = int(row["label"])
            if label is not None and row_label != label:
                continue
            if matched != index:
                matched += 1
                continue

            if "enhancer" in row and "promoter" in row:
                enhancer = row["enhancer"].strip().upper()
                promoter = row["promoter"].strip().upper()
                seq = enhancer + promoter
                anchor = len(enhancer)
            else:
                seq = (row.get("sequence") or row.get("seq") or "").strip().upper()
                anchor = None
            if not seq:
                raise ValueError(f"Selected EPI row in {path} has an empty sequence")
            crop, crop_start = _crop_sequence(seq, crop_bp, crop_mode, anchor)
            return crop, {
                "source": path,
                "label": row_label,
                "row_index_within_label": index,
                "full_length": len(seq),
                "crop_bp": crop_bp,
                "crop_mode": crop_mode,
                "crop_start": crop_start,
                "anchor": anchor,
                "junction_in_crop": None
                if anchor is None
                else max(0, min(anchor - crop_start, len(crop))),
            }
    raise ValueError(f"No EPI row found in {path} for label={label}, index={index}")


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


def token_signatures(
    tokenizer,
    sequences: list[str],
    max_length: int,
    batch_size: int,
) -> tuple[list[tuple[int, ...]], list[bool]]:
    """Return non-padded token ids and whether each sequence hit max_length."""
    ensure_padding(tokenizer)
    signatures: list[tuple[int, ...]] = []
    hit_max_length: list[bool] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                for row in input_ids:
                    ids = tuple(int(x) for x in row.tolist())
                    signatures.append(ids)
                    hit_max_length.append(len(ids) >= max_length)
            else:
                keep = input_ids != pad_token_id
                for row, row_keep in zip(input_ids, keep):
                    ids = tuple(int(x) for x in row[row_keep].tolist())
                    signatures.append(ids)
                    hit_max_length.append(len(ids) >= max_length)
        else:
            keep = attention_mask.bool()
            for row, row_keep in zip(input_ids, keep):
                ids = tuple(int(x) for x in row[row_keep].tolist())
                signatures.append(ids)
                hit_max_length.append(len(ids) >= max_length)
    return signatures, hit_max_length


def attention_mask_or_default(tokenizer, input_ids: torch.Tensor, enc: dict[str, torch.Tensor]):
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        return attention_mask
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return (input_ids != pad_token_id).long()


def call_model_with_supported_kwargs(model, kwargs: dict):
    supported = getattr(model, "_cached_forward_supported_kwargs", None)
    if supported is None:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            supported = False
        else:
            params = signature.parameters
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            supported = None if accepts_kwargs else frozenset(params)
        setattr(model, "_cached_forward_supported_kwargs", supported)

    if supported is None:
        return model(**kwargs)
    if supported is False:
        try:
            return model(**kwargs)
        except TypeError:
            fallback = {k: v for k, v in kwargs.items() if k not in ("use_cache", "return_dict")}
            return model(**fallback)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return model(**filtered)


def _evo_backbone_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    embedding_layer = getattr(backbone, "embedding_layer", None)
    if embedding_layer is None or not hasattr(embedding_layer, "embed"):
        return None
    if not hasattr(backbone, "stateless_forward"):
        return None
    hidden = backbone.embedding_layer.embed(input_ids)
    hidden, _ = backbone.stateless_forward(hidden, padding_mask=attention_mask)
    if backbone.norm is not None:
        hidden = backbone.norm(hidden)
    return hidden


def forward_hidden(model, enc: dict[str, torch.Tensor], tokenizer):
    input_ids = enc["input_ids"]
    attention_mask = attention_mask_or_default(tokenizer, input_ids, enc)

    evo_hidden = _evo_backbone_hidden(model, input_ids, attention_mask)
    if evo_hidden is not None:
        return evo_hidden

    if hasattr(model, "transformer"):
        out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    else:
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = False
        kwargs["output_hidden_states"] = True
        out = call_model_with_supported_kwargs(model, kwargs)

    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    elif hasattr(out, "hidden_states") and out.hidden_states is not None:
        hidden = out.hidden_states[-1]
    elif isinstance(out, tuple) and torch.is_tensor(out[0]) and out[0].dim() == 3:
        hidden = out[0]
    else:
        raise TypeError(f"Unsupported model output type {type(out)!r}; no hidden states found")

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is not None and hidden.shape[-1] == vocab_size:
        raise TypeError(
            "Model output appears to be logits, not hidden states "
            f"(last dim equals vocab_size={vocab_size})."
        )
    if hidden.dim() != 3:
        raise TypeError(f"Expected hidden states with shape [B, L, D], got {tuple(hidden.shape)}")
    return hidden


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
        hidden = forward_hidden(model, enc, tokenizer)
        attention_mask = attention_mask_or_default(tokenizer, enc["input_ids"], enc)
        pooled = step4.pool_hidden_states(
            hidden=hidden,
            attention_mask=attention_mask,
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


def build_mutants(sequence: str, positions: list[int], alphabet: tuple[str, ...]):
    mutants: list[str] = []
    metadata: list[tuple[int, str, str]] = []
    for pos in positions:
        original = sequence[pos].upper()
        for base in alphabet:
            if base.upper() == original:
                continue
            mutants.append(mutate_base(sequence, pos, base))
            metadata.append((pos, original, base.upper()))
    return mutants, metadata


def delta_to_contact(delta: np.ndarray, metric: str, diag: str, apc: bool):
    """delta shape: [L, A, D], with NaNs for skipped substitutions."""
    norms_raw = np.linalg.norm(np.where(np.isfinite(delta), delta, 0.0), axis=2)
    valid = np.isfinite(delta).all(axis=2) & (norms_raw > 1e-12)
    filled = np.where(np.isfinite(delta), delta, 0.0)
    num_pos = delta.shape[0]

    if metric == "effect_norm":
        scores = np.sqrt(np.square(filled).sum(axis=2))
        counts = valid.sum(axis=1).clip(min=1)
        effect = scores.sum(axis=1) / counts
        contact = np.sqrt(effect[:, None] * effect[None, :]).astype(np.float32)
    elif metric == "dot":
        weighted = filled * valid[..., None]
        contact = np.einsum("iad,jad->ij", weighted, weighted, optimize=True)
        counts = (valid[:, None, :] & valid[None, :, :]).sum(axis=2).clip(min=1)
        contact = (contact / counts).astype(np.float32)
    else:
        norms = np.linalg.norm(filled, axis=2, keepdims=True)
        unit = filled / np.clip(norms, 1e-12, None)
        unit = unit * valid[..., None]
        contact = np.einsum("iad,jad->ij", unit, unit, optimize=True)
        counts = (valid[:, None, :] & valid[None, :, :]).sum(axis=2).clip(min=1)
        contact = (contact / counts).astype(np.float32)

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
    vmax = np.nanpercentile(contact, 99) if np.isfinite(contact).any() else 1.0
    vmin = np.nanmin(contact) if np.isfinite(contact).any() else 0.0
    im = ax.imshow(contact, cmap="Blues", vmin=float(vmin), vmax=float(vmax))
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
        choices=("generic", "dnagpt", "maskedlm", "dnabert2", "caduceus", "evo", "hyena", "hyena_gated"),
        default="generic",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence")
    src.add_argument("--fasta", nargs="+")
    src.add_argument("--epi-csv", help="CSV with enhancer,promoter,label columns")
    p.add_argument("--epi-label", type=int, choices=(0, 1), default=None)
    p.add_argument("--epi-index", type=int, default=0, help="0-based row index after optional label filtering")
    p.add_argument("--epi-crop-bp", type=int, default=1024)
    p.add_argument("--epi-crop-mode", choices=("center", "junction"), default="junction")
    p.add_argument("--alphabet", nargs="+", default=list(DNA_BASES))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mutant-batch-size", type=int, default=16)
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--start-bp", type=int, default=1, help="1-based inclusive start position")
    p.add_argument("--end-bp", type=int, default=None, help="1-based inclusive end position")
    p.add_argument("--pooling", choices=("mean", "weighted_mean", "last", "cls", "eos"), default="mean")
    p.add_argument("--metric", choices=("cosine", "dot", "effect_norm"), default="cosine")
    p.add_argument("--diag", choices=("remove", "keep"), default="remove")
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


def output_prefix_for_fasta(base_prefix: str, fasta_path: str, model_name: str, multi_fasta: bool):
    if not multi_fasta:
        return base_prefix
    stem = Path(fasta_path).stem
    if base_prefix.endswith(("/", os.sep)) or (os.path.isdir(base_prefix) and not os.path.splitext(base_prefix)[1]):
        return os.path.join(base_prefix, f"{model_name}_{stem}")
    return f"{base_prefix}_{stem}"


def run_one(args, sequence: str, model, tokenizer, spec, device: str, source_meta: dict | None = None, suffix: str = ""):
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
    mutants, metadata = build_mutants(sequence, positions, alphabet)
    candidate_positions = list(positions)
    candidate_mutants = len(mutants)
    base_signatures, base_hit_max_length = token_signatures(tokenizer, [sequence], args.max_length, 1)
    base_signature = base_signatures[0]
    mutant_signatures, mutant_hit_max_length = token_signatures(
        tokenizer, mutants, args.max_length, args.mutant_batch_size
    )
    mutant_token_lengths = np.array([len(signature) for signature in mutant_signatures], dtype=np.int32)
    base_token_length = len(base_signature)
    mutants_hitting_max_length = int(sum(mutant_hit_max_length))
    print(
        "Token length after truncation:",
        f"base={base_token_length}/{args.max_length}",
        f"mutants min/median/max={int(mutant_token_lengths.min())}/"
        f"{float(np.median(mutant_token_lengths)):.1f}/{int(mutant_token_lengths.max())}",
    )
    if base_hit_max_length[0]:
        print(
            "WARNING: base sequence reaches max_length after tokenization; "
            "right-edge bp mutations may be invisible after truncation."
        )
    if mutants_hitting_max_length:
        print(
            "WARNING: mutants reaching max_length after tokenization:",
            f"{mutants_hitting_max_length}/{len(mutants)}",
            "(BPE boundary changes may create additional truncation).",
        )
    visible_pairs = [
        (mutant, item)
        for mutant, item, signature in zip(mutants, metadata, mutant_signatures)
        if signature != base_signature
    ]
    skipped_token_identical = len(mutants) - len(visible_pairs)
    visible_pos_set = {item[0] for _mutant, item in visible_pairs}
    positions = [pos for pos in positions if pos in visible_pos_set]
    skipped_token_identical_positions = len(candidate_positions) - len(positions)
    if skipped_token_identical:
        print(
            "Skipped token-identical mutants after truncation:",
            f"{skipped_token_identical}/{len(mutants)}",
        )
    if skipped_token_identical_positions:
        print(
            "Removed bp positions with no token-visible substitutions:",
            f"{skipped_token_identical_positions}/{len(candidate_positions)}",
        )
    if not visible_pairs:
        raise ValueError(
            "All mutants tokenize identically to the base sequence after truncation. "
            "Reduce the selected bp range, increase --max-length, or crop the sequence "
            "before running bp mutagenesis."
        )
    if not positions:
        raise ValueError(
            "No selected bp positions had token-visible substitutions after truncation. "
            "Reduce the selected bp range, increase --max-length, or crop the sequence "
            "before running bp mutagenesis."
        )
    mutants, metadata = map(list, zip(*visible_pairs))
    mutant_emb = encode_pooled(
        model, tokenizer, mutants, args.max_length, args.mutant_batch_size, device, args.pooling
    )

    delta = np.full((len(positions), len(alphabet), base_emb.shape[0]), np.nan, dtype=np.float32)
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
    base_to_idx = {base: idx for idx, base in enumerate(alphabet)}
    for emb, (pos, _original, new_base) in zip(mutant_emb, metadata):
        delta[pos_to_idx[pos], base_to_idx[new_base]] = emb - base_emb

    finite_delta = delta[np.isfinite(delta).all(axis=2)]
    delta_norms = np.linalg.norm(finite_delta, axis=1) if finite_delta.size else np.array([], dtype=np.float32)
    base_embedding_norm = float(np.linalg.norm(base_emb))
    delta_norm_summary = {
        "delta_norm_min": float(np.min(delta_norms)) if delta_norms.size else float("nan"),
        "delta_norm_median": float(np.median(delta_norms)) if delta_norms.size else float("nan"),
        "delta_norm_mean": float(np.mean(delta_norms)) if delta_norms.size else float("nan"),
        "delta_norm_p95": float(np.percentile(delta_norms, 95)) if delta_norms.size else float("nan"),
        "delta_norm_max": float(np.max(delta_norms)) if delta_norms.size else float("nan"),
    }
    print(
        "Embedding sensitivity:",
        f"base_norm={base_embedding_norm:.4g}",
        f"delta_norm median/p95={delta_norm_summary['delta_norm_median']:.4g}/"
        f"{delta_norm_summary['delta_norm_p95']:.4g}",
    )

    contact = delta_to_contact(delta, metric=args.metric, diag=args.diag, apc=not args.no_apc)
    prefix = args.output_prefix + suffix
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.savez_compressed(
        f"{prefix}.npz",
        delta=delta,
        contact=contact,
        positions=np.array(positions, dtype=np.int32),
        alphabet=np.array(alphabet, dtype=object),
        sequence=np.array(sequence, dtype=object),
        skipped_token_identical=np.array(skipped_token_identical, dtype=np.int32),
        skipped_token_identical_positions=np.array(skipped_token_identical_positions, dtype=np.int32),
        base_token_length=np.array(base_token_length, dtype=np.int32),
        mutant_token_lengths=mutant_token_lengths,
        mutants_hitting_max_length=np.array(mutants_hitting_max_length, dtype=np.int32),
        contact_apc_applied=np.array(not args.no_apc, dtype=np.bool_),
        base_embedding_norm=np.array(base_embedding_norm, dtype=np.float32),
        delta_norms=delta_norms.astype(np.float32),
    )
    save_contact_csv(f"{prefix}.csv", contact, positions, sequence)
    save_contact_png(f"{prefix}.png", contact, f"{spec.name} bp mutagenesis {args.metric}")
    meta = {
        "model": asdict(spec),
        "loader": args.loader,
        "sequence_length_bp": len(sequence),
        "selected_positions": len(positions),
        "alphabet": alphabet,
        "pooling": args.pooling,
        "metric": args.metric,
        "diag": args.diag,
        "contact_apc_applied": not args.no_apc,
        "base_embedding_norm": base_embedding_norm,
        **delta_norm_summary,
        "max_length": args.max_length,
        "base_token_length": base_token_length,
        "base_hit_max_length": bool(base_hit_max_length[0]),
        "mutant_token_length_min": int(mutant_token_lengths.min()),
        "mutant_token_length_median": float(np.median(mutant_token_lengths)),
        "mutant_token_length_max": int(mutant_token_lengths.max()),
        "mutants_hitting_max_length": mutants_hitting_max_length,
        "candidate_positions": len(candidate_positions),
        "evaluated_positions": len(positions),
        "skipped_token_identical_positions": skipped_token_identical_positions,
        "candidate_mutants": candidate_mutants,
        "evaluated_mutants": len(metadata),
        "skipped_token_identical": skipped_token_identical,
    }
    if source_meta:
        meta["source"] = source_meta
    with open(f"{prefix}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {prefix}.npz/.csv/.png/.meta.json")
    return contact, positions


def run_rc_consistency(args, sequence: str, contact: np.ndarray, positions: list[int], model, tokenizer, spec, device: str, source_meta: dict | None = None):
    rc_sequence = reverse_complement(sequence)
    rc_meta = dict(source_meta or {})
    rc_meta["source_type"] = f"{rc_meta.get('source_type', 'sequence')}_reverse_complement"
    rc_contact, rc_positions = run_one(
        args,
        rc_sequence,
        model,
        tokenizer,
        spec,
        device,
        source_meta=rc_meta,
        suffix="_rc",
    )
    rc_pos_to_idx = {pos: idx for idx, pos in enumerate(rc_positions)}
    aligned = [
        (fwd_idx, rc_pos_to_idx[len(sequence) - 1 - pos], pos)
        for fwd_idx, pos in enumerate(positions)
        if (len(sequence) - 1 - pos) in rc_pos_to_idx
    ]
    if len(aligned) < 2:
        raise ValueError(
            "RC consistency has fewer than two coordinate-aligned positions after "
            "token-visible filtering. Increase --max-length or use a shorter window."
        )
    fwd_idx = np.array([item[0] for item in aligned], dtype=np.int32)
    rc_idx = np.array([item[1] for item in aligned], dtype=np.int32)
    aligned_positions = np.array([item[2] for item in aligned], dtype=np.int32)
    contact_aligned = contact[np.ix_(fwd_idx, fwd_idx)]
    rc_flipped_aligned = rc_contact[np.ix_(rc_idx, rc_idx)]
    corr = float(np.corrcoef(contact_aligned.ravel(), rc_flipped_aligned.ravel())[0, 1])
    np.savez_compressed(
        f"{args.output_prefix}_rc_compare.npz",
        contact_aligned=contact_aligned,
        rc_flipped_aligned=rc_flipped_aligned,
        forward_positions=aligned_positions,
        rc_positions=np.array([len(sequence) - 1 - pos for pos in aligned_positions], dtype=np.int32),
        corr=corr,
    )
    print(
        "RC contact correlation:",
        f"{corr:.4f}",
        f"(aligned positions {len(aligned)}/{len(positions)} forward, {len(rc_positions)} rc)",
    )


def main():
    args = parse_args()
    if any(len(base) != 1 for base in args.alphabet):
        raise ValueError(f"--alphabet entries must be single characters, got {args.alphabet}")
    if args.rc_consistency and args.max_positions is not None:
        raise ValueError(
            "--rc-consistency with --max-positions is not coordinate-aligned. "
            "Remove --max-positions or run explicit matched windows."
        )
    if args.rc_consistency and (args.start_bp != 1 or args.end_bp is not None):
        raise ValueError(
            "--rc-consistency with --start-bp/--end-bp is not currently "
            "coordinate-aligned. Use the full selected sequence for RC checks."
        )
    source_meta = {}
    if args.sequence is not None:
        sequence = args.sequence.strip()
    elif args.fasta is not None:
        sequence = _read_fasta(args.fasta[0]).strip()
        source_meta = {"source": args.fasta[0], "source_type": "fasta"}
    else:
        sequence, source_meta = _read_epi_csv(
            args.epi_csv,
            label=args.epi_label,
            index=args.epi_index,
            crop_bp=args.epi_crop_bp,
            crop_mode=args.epi_crop_mode,
        )
        print(
            "Selected EPI sequence:",
            f"label={source_meta['label']}",
            f"full_length={source_meta['full_length']}",
            f"crop={len(sequence)}",
            f"junction_in_crop={source_meta['junction_in_crop']}",
        )
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if args.fp32 or device == "cpu" else torch.bfloat16
    spec = step4.ModelSpec.parse(args.model)
    model, tokenizer = _load_wrapper(args.loader)(spec, device, dtype)
    ensure_padding(tokenizer)

    if args.fasta is not None and len(args.fasta) > 1:
        original_output_prefix = args.output_prefix
        for idx, fasta_path in enumerate(args.fasta, start=1):
            print("=" * 72)
            print(f"[{idx}/{len(args.fasta)}] FASTA: {fasta_path}")
            args.output_prefix = output_prefix_for_fasta(
                original_output_prefix,
                fasta_path,
                spec.name,
                multi_fasta=True,
            )
            sequence = _read_fasta(fasta_path).strip()
            source_meta = {"source": fasta_path, "source_type": "fasta"}
            contact, positions = run_one(args, sequence, model, tokenizer, spec, device, source_meta=source_meta)
            if args.rc_consistency:
                run_rc_consistency(args, sequence, contact, positions, model, tokenizer, spec, device, source_meta=source_meta)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        args.output_prefix = original_output_prefix
        return

    contact, positions = run_one(args, sequence, model, tokenizer, spec, device, source_meta=source_meta)

    if args.rc_consistency:
        run_rc_consistency(args, sequence, contact, positions, model, tokenizer, spec, device, source_meta=source_meta)


if __name__ == "__main__":
    main()
