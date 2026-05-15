"""
Evo-specific wrapper for Step 4 frozen linear-probe evaluation.

This keeps Evo isolated from the main DNA-LLM2Vec evaluation path. Evo is a
remote-code causal LM based on StripedHyena, so we load it with
AutoModelForCausalLM and reuse the benchmark loading, pooling, and linear-probe
logic from src/step4_evaluate.py.

Usage
-----
  uv run python src/evo_baseline/step4_evo_evaluate.py \
      --models "Evo7B:togethercomputer/evo-1-131k-base:causal" \
      --gue-plus-only \
      --gue-plus-dir ./data/GUE_plus \
      --epi-crop-mode junction \
      --epi-crop-bp 8192 \
      --max-length 8192 \
      --pooling mean \
      --batch-size 1 \
      --filter-n \
      --output ./eval_results/step4_evo7b_gueplus_epi_s42.json

Notes
-----
- The Hugging Face model card recommends revision="1.1_fix"; this wrapper uses
  that revision by default. Override with EVO_REVISION=<revision> if needed.
- Evo can be memory-heavy. Start with batch-size 1 and increase only after a
  successful smoke test.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
import step4_evaluate as base  # noqa: E402


_generic_load_model = base.load_model


_EVO_REPO_PREFIXES = ("togethercomputer/evo-",)
_EVO_LOCAL_PREFIXES = ("evo-1-", "evo2-")


def _is_evo_spec(spec) -> bool:
    path = spec.path.replace("\\", "/").lower().rstrip("/")
    basename = path.rsplit("/", 1)[-1]
    return path.startswith(_EVO_REPO_PREFIXES) or basename.startswith(_EVO_LOCAL_PREFIXES)


def load_model(spec, device: str, dtype):
    """Evo loader; delegate non-Evo specs to the generic Step 4 loader."""
    if not _is_evo_spec(spec):
        return _generic_load_model(spec, device, dtype)

    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    revision = os.environ.get("EVO_REVISION", "1.1_fix")
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode}, evo-wrapper)")
    print(f"    Revision  : {revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        revision=revision,
    )
    if tokenizer.pad_token is None:
        # Some Evo remote-code revisions expose a ByteTokenizer without special
        # tokens. Reuse an existing byte token for padding so the embedding
        # matrix does not need to be resized.
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else " "
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = " "

    config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        revision=revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=dtype,
    )
    model = model.to(device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"    Parameters : {n_params:.2f}B  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


def _extract_hidden_states(out):
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, tuple):
        if len(out) >= 2 and isinstance(out[-1], (tuple, list)):
            return out[-1][-1]
        if torch.is_tensor(out[0]) and out[0].dim() == 3:
            return out[0]
    raise TypeError(
        f"Unsupported Evo output type {type(out)!r}; expected hidden_states, "
        "last_hidden_state, or tuple output."
    )


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
    """Encode sequences with Evo and return L2-normalised pooled embeddings."""
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
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    attention_mask = (input_ids != pad_token_id).long()

            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
            except TypeError as e:
                if "attention_mask" not in str(e):
                    raise
                out = model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )

            hidden = _extract_hidden_states(out)
            pooled = base.pool_hidden_states(
                hidden=hidden,
                attention_mask=attention_mask,
                input_ids=input_ids,
                pooling=pooling,
                eos_token_id=tokenizer.eos_token_id,
            )
            pooled = F.normalize(pooled, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())
        del enc, input_ids, attention_mask, out, hidden, pooled
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


base.load_model = load_model
base.encode_sequences = encode_sequences


if __name__ == "__main__":
    base.main()
