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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = str(Path(__file__).resolve().parents[1])
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
        # tokens. Choose a token already present in the tokenizer so the
        # embedding matrix does not need to be resized.
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            _set_existing_byte_pad_token(tokenizer)
    if tokenizer.pad_token_id is None:
        _set_existing_byte_pad_token(tokenizer)

    config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        revision=revision,
    )
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=dtype,
    )
    public_hidden_mode = os.environ.get("EVO_PUBLIC_HIDDEN", "auto").lower()
    if public_hidden_mode in {"0", "false", "no", "off"}:
        model._evo_public_hidden_available = False
    elif public_hidden_mode in {"1", "true", "yes", "on"}:
        model._evo_public_hidden_available = True
    model = model.to(device=device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"    Parameters : {n_params:.2f}B  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


def _set_existing_byte_pad_token(tokenizer):
    """Set padding to an existing byte-level token without resizing embeddings."""
    for candidate in (" ", "\n", "\t", "\x00", "\xff"):
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk_id:
            tokenizer.pad_token = candidate
            if tokenizer.pad_token_id is not None:
                return
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and (unk_id is None or ids[0] != unk_id):
            tokenizer.pad_token = tokenizer.decode(ids)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = int(ids[0])
            return
    raise ValueError(
        "Could not choose a pad token already present in the Evo tokenizer. "
        "Avoid adding a new special token unless the model embeddings are resized."
    )


def _prepare_evo_batch(tokenizer, batch: list[str], max_length: int, device: str):
    """Use Evo's own batching helper when the evo package is available."""
    try:
        from evo.scoring import prepare_batch
    except ImportError:
        if not getattr(_prepare_evo_batch, "_warned_missing_evo", False):
            print("[evo] using HF tokenizer path (evo package not installed)")
            _prepare_evo_batch._warned_missing_evo = True
        return None

    input_ids, seq_lengths = prepare_batch(
        batch,
        tokenizer,
        prepend_bos=False,
        device=device,
    )
    if not torch.is_tensor(seq_lengths):
        seq_lengths = torch.as_tensor(seq_lengths, device=device)
    else:
        seq_lengths = seq_lengths.to(device=device)

    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]
        seq_lengths = seq_lengths.clamp(max=max_length)

    offsets = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    attention_mask = (offsets < seq_lengths.unsqueeze(1)).long()
    return input_ids, attention_mask


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


def _evo_backbone_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """Return Evo's final sequence states before the tied vocab projection."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    missing = [
        name
        for name in ("embedding_layer", "stateless_forward", "norm")
        if not hasattr(backbone, name)
    ]
    if missing:
        revision = os.environ.get("EVO_REVISION", "1.1_fix")
        raise AttributeError(
            "This Evo wrapper expects StripedHyena remote-code internals from "
            f"EVO_REVISION={revision!r}, but backbone is missing: {missing}."
        )
    if not hasattr(backbone.embedding_layer, "embed"):
        revision = os.environ.get("EVO_REVISION", "1.1_fix")
        raise AttributeError(
            "This Evo wrapper expects backbone.embedding_layer.embed from "
            f"EVO_REVISION={revision!r}."
        )

    hidden = backbone.embedding_layer.embed(input_ids)
    hidden, _ = backbone.stateless_forward(hidden, padding_mask=attention_mask)
    if backbone.norm is not None:
        hidden = backbone.norm(hidden)
    return hidden


def _evo_public_hidden(model, input_ids: torch.Tensor):
    """Try the public HF forward path before touching StripedHyena internals."""
    if getattr(model, "_evo_public_hidden_available", None) is False:
        return None, None

    try:
        out = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    except TypeError as e:
        message = str(e)
        if "output_hidden_states" not in message and "use_cache" not in message:
            raise
        if not getattr(model, "_evo_public_hidden_warned", False):
            print("[evo] public forward does not expose hidden states; using backbone path")
            model._evo_public_hidden_warned = True
        model._evo_public_hidden_available = False
        return None, None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        model._evo_public_hidden_available = True
        return out.hidden_states[-1], out
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        model._evo_public_hidden_available = True
        return out.last_hidden_state, out
    if not getattr(model, "_evo_public_hidden_warned", False):
        print("[evo] public forward did not return hidden states; using backbone path")
        model._evo_public_hidden_warned = True
    model._evo_public_hidden_available = False
    return None, out


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
            out = None
            enc = None
            prepared = _prepare_evo_batch(tokenizer, batch, max_length, device)
            if prepared is None:
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
            else:
                input_ids, attention_mask = prepared

            hidden, out = _evo_public_hidden(model, input_ids)
            if hidden is None:
                try:
                    hidden = _evo_backbone_hidden(model, input_ids, attention_mask)
                except TypeError as e:
                    message = str(e)
                    if "attention_mask" not in message and "padding_mask" not in message:
                        raise
                    hidden = _evo_backbone_hidden(model, input_ids, None)
            if hidden is None:
                raise RuntimeError(
                    "Could not extract Evo hidden states from public forward or backbone. "
                    "Try EVO_PUBLIC_HIDDEN=0 to force the backbone path, or verify the "
                    "model revision exposes StripedHyena backbone internals."
                )
            hidden = hidden.float()
            pooled = base.pool_hidden_states(
                hidden=hidden,
                attention_mask=attention_mask,
                input_ids=input_ids,
                pooling=pooling,
                eos_token_id=tokenizer.eos_token_id,
            )
            pooled = F.normalize(pooled, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())
        del enc, input_ids, attention_mask, hidden, pooled
        if out is not None:
            del out
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    base.load_model = load_model
    base.encode_sequences = encode_sequences
    base.main()
