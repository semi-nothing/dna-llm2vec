"""
Categorical Jacobian maps for DNA foundation models.

This adapts the gLM2 categorical-Jacobian idea to DNA models. It perturbs one
token position at a time over a DNA alphabet, measures how model outputs change
at every token position, and writes an L x L contact-style map.

Two scoring modes are supported:
  - logits: compare nucleotide logits. Closest to the original gLM2 notebook.
  - hidden: compare final hidden states. Useful for encoder/embedding models.

Model spec format matches Step 4:
  name:path:mode, where mode is causal | bidir | encoder.

Examples
--------
  python src/categorical_jacobian_dna.py \
      --model "M2:./mntp_dnagpt:bidir" \
      --sequence ACGTACGTACGTACGT \
      --score-mode hidden \
      --output-prefix ./figures/jac_m2_demo

  python src/categorical_jacobian_dna.py \
      --loader dnabert2 \
      --model "DNABERT2:zhihan1996/DNABERT-2-117M:encoder" \
      --sequence ACGTACGTACGTACGT \
      --score-mode hidden \
      --output-prefix ./figures/jac_dnabert2_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import step4_evaluate as step4  # noqa: E402


DNA_BASES = ("A", "C", "G", "T")


def _load_wrapper(name: str):
    if name == "generic":
        return step4.load_model
    if name == "maskedlm":
        return load_masked_lm
    if name == "dnabert2":
        import step4_dnabert2_evaluate as wrapper

        return wrapper.load_model
    if name == "caduceus":
        from caduceus_baseline import step4_caduceus_evaluate as wrapper

        return wrapper.load_model
    if name == "evo":
        from evo_baseline import step4_evo_evaluate as wrapper

        return wrapper.load_model
    raise ValueError(f"Unknown loader {name!r}")


def load_masked_lm(spec, device: str, dtype):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, masked-lm loader)")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ensure_padding(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


def _read_fasta(path: str) -> str:
    parts: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts.append(line)
    return "".join(parts)


def _single_token_id(tokenizer, base: str) -> int | None:
    candidates = (base, base.lower())
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for candidate in candidates:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if token_id is not None and token_id != unk_id:
            return int(token_id)

    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and (unk_id is None or ids[0] != unk_id):
            return int(ids[0])

    return None


def dna_token_ids(tokenizer, bases: Iterable[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    labels: list[str] = []
    missing: list[str] = []
    for base in bases:
        token_id = _single_token_id(tokenizer, base)
        if token_id is None:
            missing.append(base)
        else:
            ids.append(token_id)
            labels.append(base)
    if missing:
        raise ValueError(
            "Tokenizer cannot represent these DNA bases as single tokens: "
            f"{missing}. Categorical token substitutions require single-token "
            "alphabet entries. For k-mer tokenizers, use token-level k-mer "
            "labels via --alphabet-token-ids or switch to hidden sequence-level "
            "mutagenesis."
        )
    return ids, labels


def ensure_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else " "
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = " "


def tokenize_sequence(tokenizer, sequence: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    ensure_padding(tokenizer)
    enc = tokenizer(
        sequence,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def token_labels(tokenizer, input_ids: torch.Tensor) -> list[str]:
    ids = input_ids.detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return [str(t) for t in tokens]


def valid_mutation_positions(input_ids: torch.Tensor, alphabet_ids: list[int]) -> list[int]:
    alphabet = set(int(x) for x in alphabet_ids)
    ids = input_ids.detach().cpu().tolist()
    return [i for i, token_id in enumerate(ids) if int(token_id) in alphabet]


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, token_ids: list[int]):
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        out = model(**kwargs, use_cache=False)
    except TypeError:
        out = model(**kwargs)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[..., token_ids].detach().cpu().float()


def _evo_backbone_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None
    hidden = backbone.embedding_layer.embed(input_ids)
    hidden, _ = backbone.stateless_forward(hidden, padding_mask=attention_mask)
    if backbone.norm is not None:
        hidden = backbone.norm(hidden)
    return hidden


def forward_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    evo_hidden = _evo_backbone_hidden(model, input_ids, attention_mask)
    if evo_hidden is not None:
        return evo_hidden.detach().cpu().float()

    if hasattr(model, "transformer"):
        out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
    else:
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        try:
            out = model(**kwargs, output_hidden_states=False, use_cache=False)
        except TypeError:
            out = model(**kwargs)

    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    elif isinstance(out, tuple):
        hidden = out[0]
    else:
        raise TypeError(f"Unsupported model output type {type(out)!r}")
    return hidden.detach().cpu().float()


def jac_to_contact(jac: np.ndarray, center: bool = True, diag: str = "remove", apc: bool = True) -> np.ndarray:
    """Convert [L, A, L, B] categorical Jacobian to an L x L contact map."""
    x = jac.copy()
    if center:
        for axis in range(4):
            if x.shape[axis] > 1:
                x -= x.mean(axis=axis, keepdims=True)

    contacts = np.sqrt(np.square(x).sum(axis=(1, 3)))
    contacts = (contacts + contacts.T) / 2.0

    if diag == "remove":
        np.fill_diagonal(contacts, 0.0)
    elif diag == "normalize":
        contacts_diag = np.diag(contacts)
        denom = np.sqrt(contacts_diag[:, None] * contacts_diag[None, :])
        contacts = contacts / np.clip(denom, 1e-9, None)

    if apc:
        total = contacts.sum()
        if abs(total) > 1e-12:
            ap = contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / total
            contacts = contacts - ap

    if diag == "remove":
        np.fill_diagonal(contacts, 0.0)
    return contacts


def hidden_delta_to_contact(delta: np.ndarray, center: bool = True, diag: str = "remove", apc: bool = True):
    """Convert [L, A, L, D] hidden deltas to an L x L contact map."""
    x = delta.copy()
    if center:
        for axis in (0, 1, 2):
            if x.shape[axis] > 1:
                x -= x.mean(axis=axis, keepdims=True)
    contacts = np.sqrt(np.square(x).sum(axis=3).mean(axis=1))
    contacts = (contacts + contacts.T) / 2.0
    if diag == "remove":
        np.fill_diagonal(contacts, 0.0)
    if apc:
        total = contacts.sum()
        if abs(total) > 1e-12:
            contacts = contacts - contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / total
    if diag == "remove":
        np.fill_diagonal(contacts, 0.0)
    return contacts


def compute_logits_jacobian(model, input_ids, attention_mask, alphabet_ids, positions, batch_size, device):
    base = forward_logits(model, input_ids, attention_mask, alphabet_ids)[0]
    seq_len = input_ids.size(1)
    num_tokens = len(alphabet_ids)
    jac = np.zeros((seq_len, num_tokens, seq_len, num_tokens), dtype=np.float32)

    for pos in tqdm(positions, desc="Categorical Jacobian"):
        mutants = input_ids.repeat(num_tokens, 1)
        mutants[:, pos] = torch.tensor(alphabet_ids, device=device, dtype=input_ids.dtype)
        mask = attention_mask.repeat(num_tokens, 1) if attention_mask is not None else None
        chunks: list[torch.Tensor] = []
        for start in range(0, num_tokens, batch_size):
            end = start + batch_size
            chunks.append(forward_logits(model, mutants[start:end], None if mask is None else mask[start:end], alphabet_ids))
        fx = torch.cat(chunks, dim=0)
        jac[pos] = (fx - base.unsqueeze(0)).numpy()
        del mutants, mask, fx
        if device == "cuda":
            torch.cuda.empty_cache()
    return jac


def compute_hidden_delta(model, input_ids, attention_mask, alphabet_ids, positions, batch_size, device):
    base = forward_hidden(model, input_ids, attention_mask)[0]
    seq_len, hidden_dim = base.shape
    num_tokens = len(alphabet_ids)
    delta = np.zeros((seq_len, num_tokens, seq_len, hidden_dim), dtype=np.float32)

    for pos in tqdm(positions, desc="Hidden delta"):
        mutants = input_ids.repeat(num_tokens, 1)
        mutants[:, pos] = torch.tensor(alphabet_ids, device=device, dtype=input_ids.dtype)
        mask = attention_mask.repeat(num_tokens, 1) if attention_mask is not None else None
        chunks: list[torch.Tensor] = []
        for start in range(0, num_tokens, batch_size):
            end = start + batch_size
            chunks.append(forward_hidden(model, mutants[start:end], None if mask is None else mask[start:end]))
        hx = torch.cat(chunks, dim=0)
        delta[pos] = (hx - base.unsqueeze(0)).numpy()
        del mutants, mask, hx
        if device == "cuda":
            torch.cuda.empty_cache()
    return delta


def save_contact_csv(path: str, contact: np.ndarray, tokens: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["i", "j", "i_token", "j_token", "value"])
        for i in range(contact.shape[0]):
            for j in range(contact.shape[1]):
                writer.writerow([i + 1, j + 1, tokens[i], tokens[j], float(contact[i, j])])


def save_contact_png(path: str, contact: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    vmax = np.percentile(contact, 99) if np.isfinite(contact).any() else 1.0
    im = ax.imshow(contact, cmap="Blues", vmin=float(np.min(contact)), vmax=float(vmax))
    ax.set_title(title)
    ax.set_xlabel("token position")
    ax.set_ylabel("token position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model spec: name:path:mode")
    p.add_argument(
        "--loader",
        choices=("generic", "maskedlm", "dnabert2", "caduceus", "evo"),
        default="generic",
        help="Optional Step4 wrapper loader for fragile remote-code models.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="DNA sequence string")
    src.add_argument("--fasta", help="FASTA file containing one sequence")
    p.add_argument("--score-mode", choices=("logits", "hidden"), default="hidden")
    p.add_argument("--alphabet", nargs="+", default=list(DNA_BASES), help="Single-token mutation alphabet")
    p.add_argument("--alphabet-token-ids", nargs="+", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mutant-batch-size", type=int, default=4)
    p.add_argument("--max-positions", type=int, default=None, help="Limit mutated positions for smoke tests")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--no-apc", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sequence = args.sequence if args.sequence is not None else _read_fasta(args.fasta)
    sequence = sequence.strip()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if args.fp32 or device == "cpu" else torch.bfloat16

    spec = step4.ModelSpec.parse(args.model)
    load_model = _load_wrapper(args.loader)
    model, tokenizer = load_model(spec, device, dtype)
    ensure_padding(tokenizer)

    if args.alphabet_token_ids is None:
        alphabet_ids, alphabet_labels = dna_token_ids(tokenizer, args.alphabet)
    else:
        alphabet_ids = args.alphabet_token_ids
        alphabet_labels = tokenizer.convert_ids_to_tokens(alphabet_ids)

    enc = tokenize_sequence(tokenizer, sequence, args.max_length, device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    tokens = token_labels(tokenizer, input_ids[0])
    positions = valid_mutation_positions(input_ids[0], alphabet_ids)
    if args.max_positions is not None:
        positions = positions[: args.max_positions]
    if not positions:
        raise ValueError(
            "No mutable positions found. This usually means the tokenizer does "
            "not represent the sequence with the chosen alphabet token ids."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    print(f"Mutable positions: {len(positions)} / {input_ids.size(1)}")
    print(f"Alphabet: {list(zip(alphabet_labels, alphabet_ids))}")

    if args.score_mode == "logits":
        jac = compute_logits_jacobian(
            model, input_ids, attention_mask, alphabet_ids, positions, args.mutant_batch_size, device
        )
        contact = jac_to_contact(jac, apc=not args.no_apc)
        np.savez_compressed(
            f"{args.output_prefix}.npz",
            jacobian=jac,
            contact=contact,
            input_ids=input_ids.detach().cpu().numpy(),
            alphabet_ids=np.array(alphabet_ids),
            tokens=np.array(tokens, dtype=object),
            positions=np.array(positions),
        )
    else:
        delta = compute_hidden_delta(
            model, input_ids, attention_mask, alphabet_ids, positions, args.mutant_batch_size, device
        )
        contact = hidden_delta_to_contact(delta, apc=not args.no_apc)
        np.savez_compressed(
            f"{args.output_prefix}.npz",
            hidden_delta=delta,
            contact=contact,
            input_ids=input_ids.detach().cpu().numpy(),
            alphabet_ids=np.array(alphabet_ids),
            tokens=np.array(tokens, dtype=object),
            positions=np.array(positions),
        )

    save_contact_csv(f"{args.output_prefix}.csv", contact, tokens)
    save_contact_png(f"{args.output_prefix}.png", contact, f"{spec.name} {args.score_mode} categorical map")
    with open(f"{args.output_prefix}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": asdict(spec),
                "loader": args.loader,
                "score_mode": args.score_mode,
                "sequence_length_bp": len(sequence),
                "token_length": int(input_ids.size(1)),
                "mutable_positions": len(positions),
                "alphabet": list(zip(alphabet_labels, alphabet_ids)),
                "max_length": args.max_length,
            },
            fh,
            indent=2,
        )
    print(f"Wrote {args.output_prefix}.npz/.csv/.png/.meta.json")


if __name__ == "__main__":
    main()
