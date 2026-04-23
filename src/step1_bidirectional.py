"""
DNA-LLM2Vec  |  Step 1: Causal → Bidirectional Attention Conversion
====================================================================
Converts DNAGPT (GPT-2 decoder) to a bidirectional encoder by replacing
the causal attention mask with a full attention mask.

GPU requirements (RTX 5090 / Blackwell sm_120):
  - CUDA  >= 12.8
  - PyTorch >= 2.6  (2.7 recommended for stable Blackwell support)
  - Use bfloat16, NOT float16  (Blackwell prefers bf16)

Usage:
  python step1_bidirectional.py
  python step1_bidirectional.py --model dnagpt/human_gpt2-v1 --output ./bidir_dnagpt
"""

import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config


# ── 0. Environment check ──────────────────────────────────────────────────────

def check_environment():
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")

    if not torch.cuda.is_available():
        print("  CUDA       : NOT available — will run on CPU (slow)")
        return "cpu"

    cuda_ver = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cap  = torch.cuda.get_device_capability(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"  CUDA       : {cuda_ver}")
    print(f"  GPU        : {gpu_name}")
    print(f"  Capability : sm_{gpu_cap[0]}{gpu_cap[1]}")
    print(f"  VRAM       : {vram_gb:.1f} GB")

    # Blackwell (sm_120) needs CUDA >= 12.8 and PyTorch >= 2.6
    major_cap = gpu_cap[0]
    if major_cap >= 12:  # Blackwell
        cuda_major = int(cuda_ver.split(".")[0])
        torch_minor = int(torch.__version__.split(".")[1])
        if cuda_major < 12:
            print("\n  [WARN] RTX 5090 needs CUDA >= 12.8. Please update.")
        if torch_minor < 6:
            print("\n  [WARN] RTX 5090 needs PyTorch >= 2.6. Please update.")

    print("=" * 60)
    return "cuda"


# ── 1. Load DNAGPT ────────────────────────────────────────────────────────────

def load_dnagpt(model_name: str, device: str):
    """
    Load DNAGPT from HuggingFace.
    Uses bfloat16 on CUDA — preferred for Blackwell (RTX 5090).
    Falls back to float32 on CPU.
    """
    print(f"\n[1/4] Loading model: {model_name}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some GPT-2 tokenizers lack a pad token; set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",   # REQUIRED: transformers>=4.36 defaults GPT-2 to
    )                                  # SDPA which has no bias buffer to patch.
                                       # eager restores the classic implementation.

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded  : {n_params:.1f}M parameters")
    print(f"  dtype         : {dtype}")
    print(f"  Architecture  : {model.config.model_type}")
    print(f"  Attn impl     : eager (bias buffer patchable)")

    return model, tokenizer


# ── 2. Bidirectional attention patch ─────────────────────────────────────────

def patch_to_bidirectional(model: nn.Module) -> nn.Module:
    """
    Disable causal masking — makes a GPT-2 decoder fully bidirectional.

    Three strategies are applied in order of precedence:

    Strategy A — config.is_causal = False  (transformers >= 5.x)
      create_causal_mask() checks config.is_causal and delegates to
      create_bidirectional_mask() when it is False.  This is the
      official supported API for bidirectional attention in 5.x.

    Strategy B — module.is_causal = False  (transformers 4.36–4.x)
      Each GPT2Attention stores self.is_causal = True which is forwarded
      to the unified eager_attention_forward / SDPA interface.

    Strategy C — bias buffer  (transformers < 4.36)
      GPT2Attention registered a lower-triangular bias buffer
      (shape 1,1,T,T).  Fill it with all-ones to disable masking.

    Note: none of these changes are saved in the state_dict.
    Re-apply this patch after every load, or use load_bidirectional_dnagpt().
    """
    print("\n[2/4] Patching causal → bidirectional attention")

    # ── Strategy A: config flag (transformers >= 5.x) ─────────────────────
    config_patched = False
    if hasattr(model, "config"):
        model.config.is_causal = False
        model.config.is_bidirectional = True   # record our patch
        config_patched = True
        print("  [config]    model.config.is_causal = False  (transformers 5.x path)")

    # ── Strategies B & C: per-layer patches (older transformers) ──────────
    patched_bias = 0
    patched_flag = 0

    for name, module in model.named_modules():
        if "Attention" not in module.__class__.__name__:
            continue

        # Strategy B: is_causal instance attribute
        if getattr(module, "is_causal", None) is True:
            module.is_causal = False
            patched_flag += 1
            print(f"  [is_causal] Patched : {name}")

        # Strategy C: 4-D lower-triangular bias buffer
        if hasattr(module, "bias"):
            bias = module.bias
            if isinstance(bias, torch.Tensor) and bias.dim() == 4:
                module.bias = torch.ones_like(bias)
                patched_bias += 1
                print(f"  [bias]      Patched : {name}  shape={list(bias.shape)}")

    if not config_patched and patched_bias == 0 and patched_flag == 0:
        raise RuntimeError(
            "patch_to_bidirectional() found nothing to patch.\n"
            "Check that the model is a GPT-2 family architecture."
        )

    print(f"\n  config.is_causal=False  : {'yes' if config_patched else 'no'}")
    print(f"  is_causal flag layers   : {patched_flag}")
    print(f"  bias buffer layers      : {patched_bias}")
    print("  Bidirectional attention : ENABLED")

    return model


# ── 3. Sanity check ──────────────────────────────────────────────────────────

def sanity_check(model_name: str, device: str, tokenizer):
    """
    Verify the patch changes hidden states as expected.

    For a causal model, token i cannot attend to token j > i.
    After patching, token i CAN attend to all tokens.

    Expected result:
      - Hidden states DIFFER between causal and bidirectional models.
      - The LAST token's hidden state should differ the most (it already
        had full left context in the causal model, so earlier tokens
        show a bigger change).
    """
    print("\n[3/4] Sanity check")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    seq = "ATGCATGCATGCATGC"  # Short DNA sequence for the test
    inputs = tokenizer(seq, return_tensors="pt").to(device)

    # --- Causal model (original) ---
    causal_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    causal_model.eval()
    with torch.no_grad():
        causal_out = causal_model(**inputs, output_hidden_states=True)
    causal_hidden = causal_out.hidden_states[-1]  # last layer, shape (1, T, D)

    # --- Bidirectional model (patched) ---
    bidir_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    bidir_model = patch_to_bidirectional(bidir_model)
    bidir_model.eval()
    with torch.no_grad():
        bidir_out = bidir_model(**inputs, output_hidden_states=True)
    bidir_hidden = bidir_out.hidden_states[-1]

    # --- Compare ---
    diff = (causal_hidden - bidir_hidden).abs()
    print(f"  Sequence          : {seq}")
    print(f"  Hidden state diff (mean) : {diff.mean().item():.6f}")
    print(f"  Hidden state diff (max)  : {diff.max().item():.6f}")

    if diff.mean().item() > 1e-5:
        print("  Result : PASS — hidden states differ as expected.")
    else:
        print("  Result : WARN — hidden states are identical.")
        print("           The patch may not have taken effect.")

    # Per-token diff to show positional effect
    per_token = diff[0].mean(dim=-1)  # shape (T,)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print("\n  Per-token mean diff:")
    for tok, d in zip(tokens, per_token.float().tolist()):
        bar = "█" * int(d * 200)
        print(f"    {tok:>10}  {d:.6f}  {bar}")

    del causal_model, bidir_model
    if device == "cuda":
        torch.cuda.empty_cache()


# ── 4. Save patched model ────────────────────────────────────────────────────

def save_patched_model(model, tokenizer, output_dir: str):
    """
    Save the patched model and tokenizer.

    Important: because the `bias` buffer is non-persistent, it is NOT
    stored in the state_dict. The output directory includes a README
    reminding you to call patch_to_bidirectional() after loading.
    Use load_bidirectional_dnagpt() to load safely.
    """
    print(f"\n[4/4] Saving patched model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Write a reminder file
    note_path = os.path.join(output_dir, "BIDIR_NOTE.txt")
    with open(note_path, "w") as f:
        f.write(
            "This model was patched by step1_bidirectional.py.\n"
            "The causal attention mask (bias buffer) is non-persistent and\n"
            "is NOT saved in the state_dict.\n\n"
            "Always load with:\n"
            "  from step1_bidirectional import load_bidirectional_dnagpt\n"
            "  model, tokenizer = load_bidirectional_dnagpt('./bidir_dnagpt')\n"
        )

    print(f"  Saved model weights  : {output_dir}/pytorch_model.bin (or shards)")
    print(f"  Saved tokenizer      : {output_dir}/tokenizer files")
    print(f"  Note                 : {note_path}")


# ── Public loader (re-applies patch after loading) ────────────────────────────

def load_bidirectional_dnagpt(model_dir: str, device: str = "cuda"):
    """
    Load a previously saved patched model and re-apply the
    bidirectional patch (since bias buffer is non-persistent).

    Example:
        model, tokenizer = load_bidirectional_dnagpt("./bidir_dnagpt")
    """
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
    )
    model = patch_to_bidirectional(model)
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DNA-LLM2Vec Step 1")
    parser.add_argument(
        "--model", default="dnagpt/human_gpt2-v1",
        help="HuggingFace model ID (default: dnagpt/human_gpt2-v1)"
    )
    parser.add_argument(
        "--output", default="./bidir_dnagpt",
        help="Output directory for the patched model"
    )
    parser.add_argument(
        "--skip-sanity", action="store_true",
        help="Skip the sanity check (faster, useful if VRAM is tight)"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--run-name", default=None,
                        help="Optional experiment/run label for command consistency.")
    parser.add_argument("--repeat-index", type=int, default=None,
                        help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 0. Environment
    device = check_environment()

    # 1. Load
    model, tokenizer = load_dnagpt(args.model, device)

    # 2. Patch
    model = patch_to_bidirectional(model)

    # 3. Sanity check
    if not args.skip_sanity:
        sanity_check(args.model, device, tokenizer)
    else:
        print("\n[3/4] Sanity check skipped.")

    # 4. Save
    save_patched_model(model, tokenizer, args.output)

    print("\nStep 1 complete.")
    print(f"Next step: run step2_mntp.py --model {args.output}")
