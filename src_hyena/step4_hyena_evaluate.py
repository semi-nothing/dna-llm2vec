"""
HyenaDNA-specific wrapper for Step 4 evaluation.

This reuses the benchmark loading, pooling, and linear-probe machinery from
`src/step4_evaluate.py`, while isolating the HyenaDNA loading path here.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
import step4_evaluate as base  # noqa: E402

from common import (  # noqa: E402
    count_parameters_m,
    ensure_attention_mask,
    extract_hidden_states,
    load_hyena_backbone,
    load_hyena_tokenizer,
)


_generic_load_model = base.load_model


def load_model(spec, device: str, dtype):
    if spec.mode != "encoder":
        return _generic_load_model(spec, device, dtype)

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, hyena-wrapper)")

    tokenizer = load_hyena_tokenizer(path)
    model, load_path = load_hyena_backbone(path, device=device)

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
    desc: str = "Encoding",
) -> np.ndarray:
    model.eval()
    all_embeddings = []

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
            pooled = F.normalize(pooled, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())
        del enc, out, hidden, pooled

    return np.concatenate(all_embeddings, axis=0)


base.load_model = load_model
base.encode_sequences = encode_sequences


if __name__ == "__main__":
    base.main()
