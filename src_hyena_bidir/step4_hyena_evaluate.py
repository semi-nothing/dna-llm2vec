"""
HyenaDNA Step 4 evaluation wrapper for the honest-bidirectional branch.

This mirrors `src_hyena/step4_hyena_evaluate.py`, but resolves `common.py`
from `src_hyena_bidir` so saved honest-bidir checkpoints are re-patched before
embedding extraction.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
CURRENT_DIR = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
for path in (CURRENT_DIR, ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import step4_evaluate as base  # noqa: E402

from common import (  # noqa: E402
    count_parameters_m,
    ensure_attention_mask,
    extract_hidden_states,
    load_hyena_backbone,
    load_hyena_tokenizer,
)


_generic_load_model = base.load_model
_generic_encode_sequences = base.encode_sequences


def load_model(spec, device: str, dtype):
    if spec.mode != "encoder":
        return _generic_load_model(spec, device, dtype)

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, honest-bidir-wrapper)")

    tokenizer = load_hyena_tokenizer(path)
    model, load_path = load_hyena_backbone(path, device=device, dtype=dtype)
    setattr(model, "_hyena_eval_wrapper", True)

    print(f"    Load path  : {load_path}")
    print(f"    Parameters : {count_parameters_m(model):.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


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
    if not getattr(model, "_hyena_eval_wrapper", False):
        return _generic_encode_sequences(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            pooling=pooling,
            normalize=normalize,
            desc=desc,
        )

    model.eval()
    all_embeddings = []

    if not sequences:
        raise ValueError(f"{desc}: no sequences to encode")

    for i in tqdm(range(0, len(sequences), batch_size), desc=desc, leave=False):
        batch = sequences[i : i + batch_size]
        with torch.inference_mode():
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding="longest",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            attention_mask = ensure_attention_mask(enc, tokenizer.pad_token_id)

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = extract_hidden_states(out)
            pooled = base.pool_hidden_states(
                hidden=hidden,
                attention_mask=attention_mask,
                input_ids=enc["input_ids"],
                pooling=pooling,
                eos_token_id=tokenizer.eos_token_id,
            )
            if normalize:
                pooled = F.normalize(pooled, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())
        del enc, out, hidden, pooled

    return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    base.load_model = load_model
    base.encode_sequences = encode_sequences
    base.main()
