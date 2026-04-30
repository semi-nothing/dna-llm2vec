"""
Caduceus-specific wrapper for Step 4 evaluation.

Why this exists
---------------
Caduceus uses a custom remote-code model stack rather than the GPT/ESM-style
loading path used by the main DNA-LLM2Vec experiments. We keep its loader
isolated here so we can evaluate it as a Step-4 linear-probe baseline without
touching the main `step4_evaluate.py` pipeline.

Design
------
- Reuse all benchmark loading, pooling, and linear-probe logic from
  `src/step4_evaluate.py`
- Override only the model loader for `encoder` mode
- Prefer `AutoModel`; fall back to `AutoModelForMaskedLM` if needed

Usage
-----
  uv run python src/caduceus_baseline/step4_caduceus_evaluate.py \
      --models "Caduceus:kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16:encoder" \
      --gb-only \
      --output ./eval_results/caduceus_gb.json
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
import step4_evaluate as base  # noqa: E402


_generic_load_model = base.load_model


def load_model(spec, device: str, dtype):
    """Caduceus-specific loader for encoder mode; delegate all else."""
    if spec.mode != "encoder":
        return _generic_load_model(spec, device, dtype)

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, caduceus-wrapper)")

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    load_errors = []

    try:
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        print("    Load path  : AutoModel")
    except Exception as e:
        load_errors.append(f"AutoModel failed: {e}")

    if model is None:
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            print("    Load path  : AutoModelForMaskedLM")
        except Exception as e:
            load_errors.append(f"AutoModelForMaskedLM failed: {e}")

    if model is None:
        raise RuntimeError(
            "Could not load Caduceus with either AutoModel or AutoModelForMaskedLM.\n"
            + "\n".join(load_errors)
        )

    model = model.to(device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")
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
    """
    Caduceus-safe encoder wrapper.

    Accepts several possible output styles from remote-code models:
    - objects with `last_hidden_state`
    - objects with `hidden_states`
    - tuples whose first element is the hidden tensor
    """
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
                    f"Unsupported Caduceus output type {type(out)!r}; "
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
