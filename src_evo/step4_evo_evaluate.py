"""
Evo Step 4: frozen linear-probe evaluation for E0--E6.

This is a src_evo-local wrapper around the shared Step 4 benchmark code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = str(Path(__file__).resolve().parents[1])
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(__file__))

import step4_evaluate as base  # noqa: E402
from common import (  # noqa: E402
    count_parameters_m,
    evo_backbone_hidden,
    evo_hidden_states,
    is_evo_32bit_index_error,
    load_evo_causal_lm,
    load_evo_tokenizer,
    prepare_evo_batch,
)

_generic_load_model = base.load_model


def _is_evo_spec(spec) -> bool:
    path = spec.path.replace("\\", "/").lower().rstrip("/")
    basename = path.rsplit("/", 1)[-1]
    return spec.mode == "evo" or path.startswith("togethercomputer/evo-") or basename.startswith(("evo_", "evo-", "evo2-"))


def load_model(spec, device: str, dtype):
    if not _is_evo_spec(spec):
        return _generic_load_model(spec, device, dtype)
    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, evo)")
    tokenizer = load_evo_tokenizer(path)
    model, load_path = load_evo_causal_lm(path, device=device, dtype=dtype)
    model.eval()
    print(f"    Load path  : {load_path}")
    print(f"    Parameters : {count_parameters_m(model):.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


def _encode_evo_batch(
    model,
    tokenizer,
    batch: list[str],
    max_length: int,
    device: str,
    pooling: str,
    normalize: bool,
) -> torch.Tensor:
    input_ids, attention_mask = prepare_evo_batch(tokenizer, batch, max_length, device)
    try:
        hidden = evo_backbone_hidden(model, input_ids, attention_mask)
        if hidden is None:
            hidden = evo_hidden_states(model, input_ids, attention_mask)
    except RuntimeError as e:
        if not is_evo_32bit_index_error(e) or len(batch) <= 1:
            raise
        del input_ids, attention_mask
        if device == "cuda":
            torch.cuda.empty_cache()
        if not getattr(_encode_evo_batch, "_warned_32bit_index", False):
            print("[evo] CUDA FIR kernel hit 32-bit indexing limit; retrying with smaller microbatches.")
            _encode_evo_batch._warned_32bit_index = True
        midpoint = len(batch) // 2
        pieces = [
            _encode_evo_batch(model, tokenizer, batch[:midpoint], max_length, device, pooling, normalize),
            _encode_evo_batch(model, tokenizer, batch[midpoint:], max_length, device, pooling, normalize),
        ]
        return torch.cat(pieces, dim=0)

    hidden = hidden.float()
    pooled = base.pool_hidden_states(
        hidden=hidden,
        attention_mask=attention_mask,
        input_ids=input_ids,
        pooling=pooling,
        eos_token_id=tokenizer.eos_token_id,
    )
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    del input_ids, attention_mask, hidden
    return pooled


def encode_sequences(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str = "mean",
    normalize: bool = True,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode sequences with Evo and return L2-normalised pooled embeddings."""
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc=desc, leave=False):
        batch = sequences[i : i + batch_size]
        with torch.inference_mode():
            pooled = _encode_evo_batch(model, tokenizer, batch, max_length, device, pooling, normalize)
        all_embeddings.append(pooled.cpu().float().numpy())
        del pooled
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    base.load_model = load_model
    base.encode_sequences = encode_sequences
    base.ModelSpec.parse = staticmethod(
        lambda spec: base.ModelSpec(*spec.rsplit(":", 2))
    )
    base.main()
