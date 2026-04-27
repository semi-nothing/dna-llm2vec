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

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import step4_evaluate as base

DNABERT2_REVISION = "refs/pr/32"


_generic_load_model = base.load_model


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

    # Conservative load path for remote-code encoder models:
    # - disable low_cpu_mem_usage to avoid meta/lazy materialization
    # - disable fast init so tensors are created eagerly
    # - load in float32 first, then cast/move after full materialization
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
    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


base.load_model = load_model


if __name__ == "__main__":
    base.main()
