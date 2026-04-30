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

from common import count_parameters_m, load_hyena_backbone, load_hyena_tokenizer  # noqa: E402


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

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )

            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                hidden = out.hidden_states[-1]
            elif isinstance(out, tuple):
                hidden = out[0]
            else:
                raise TypeError(
                    f"Unsupported HyenaDNA output type {type(out)!r}; "
                    "expected last_hidden_state, hidden_states, or tuple output."
                )

            pooled = base.pool_hidden_states(
                hidden=hidden,
                attention_mask=enc["attention_mask"],
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
