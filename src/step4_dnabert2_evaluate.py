"""
DNABERT-2-specific wrapper for Step 4 evaluation.

Why this exists
---------------
DNABERT-2 uses a remote-code encoder implementation that can be fragile under
newer Transformers / PyTorch combinations when loaded through the generic
step4_evaluate.py path. In particular, model construction may fail with:

    RuntimeError: Tensor on device meta is not on the expected device cpu!

This wrapper keeps the exact same Step 4 evaluation flow, but overrides the
encoder loader with a more conservative DNABERT-2-specific loading routine.

Usage
-----
Use the same CLI as step4_evaluate.py. Example:

  uv run python src/step4_dnabert2_evaluate.py \
      --models "D0:zhihan1996/DNABERT-2-117M:encoder" \
      --gb-only \
      --output ./eval_results/dnabert2_gb.json
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

import step4_evaluate as base

DNABERT2_REVISION = "refs/pr/32"


_generic_load_model = base.load_model
_generic_encode_sequences = base.encode_sequences


def load_model(spec, device: str, dtype):
    """DNABERT-2-specific loader for encoder mode; delegate all else."""
    if spec.mode != "encoder":
        return _generic_load_model(spec, device, dtype)

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, dnabert2-wrapper)")

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        revision=DNABERT2_REVISION,
        force_download=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc_config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        revision=DNABERT2_REVISION,
        force_download=True,
    )
    if not hasattr(enc_config, "pad_token_id") or enc_config.pad_token_id is None:
        enc_config.pad_token_id = tokenizer.pad_token_id or 0
    if hasattr(enc_config, "use_cache"):
        enc_config.use_cache = False
    # DNABERT-2's custom Bert layers use Triton flash attention only when
    # attention_probs_dropout_prob == 0.0. A tiny non-zero value forces the
    # PyTorch attention path, which is much more robust across Triton versions.
    if hasattr(enc_config, "attention_probs_dropout_prob"):
        enc_config.attention_probs_dropout_prob = max(
            float(getattr(enc_config, "attention_probs_dropout_prob", 0.0)),
            1e-6,
        )

    # Conservative load path for remote-code encoder models:
    # - disable low_cpu_mem_usage to avoid meta/lazy materialization
    # - disable fast init so tensors are created eagerly
    # - keep the whole model in float32 even on GPU
    #
    # DNABERT-2's non-flash attention path mixes internal tensors that remain
    # float32 with attention value tensors that would become bf16/fp16 if we
    # cast the module to the caller-provided dtype. That later crashes with:
    #   RuntimeError: expected scalar type BFloat16 but found Float
    #
    # For this wrapper, robustness matters more than memory savings, so we keep
    # DNABERT-2 in fp32 end-to-end.
    model = AutoModel.from_pretrained(
        path,
        config=enc_config,
        trust_remote_code=True,
        revision=DNABERT2_REVISION,
        force_download=True,
        low_cpu_mem_usage=False,
        _fast_init=False,
        torch_dtype=torch.float32,
        device_map=None,
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
    """DNABERT-2-safe encoder wrapper.

    DNABERT-2 remote code may return a tuple instead of a HF output object.
    Keep the generic pooling path but accept both output formats.
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

            if hasattr(model, "transformer"):
                out = model.transformer(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
            else:
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )

            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif isinstance(out, tuple):
                hidden = out[0]
            else:
                raise TypeError(
                    f"Unsupported model output type {type(out)!r}; "
                    "expected an object with last_hidden_state or a tuple whose "
                    "first element is hidden states."
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
