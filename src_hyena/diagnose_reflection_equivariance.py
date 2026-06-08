"""
Reflection-equivariance diagnostic for HyenaDNA bidirectional variants.

Goal
----
Test directly whether a checkpoint's backbone is *reflection equivariant*, i.e.
whether reversing the input (pure positional reversal, NOT reverse complement)
just reverses the per-token hidden states:

        H(reverse(x))  ==  reverse(H(x))   ?

If this holds AND the readout is mean pooling (order-invariant), then the pooled
embedding satisfies f(x) == f(reverse(x)) and the model is structurally blind to
5'->3' orientation. That is exactly the failure mode we suspect for the old
"tied + fixed 0.5 average" bidirectional patch, and it should be visibly broken
by the untied two-branch (`bidir`) and the learned-gate (`gated`) variants once
their direction-specific parameters have trained.

Two metrics per sequence
------------------------
1. token_equivariance_cos : mean over tokens of
       cos( reverse(H(x))[t] , H(reverse(x))[t] )
   Computed pre-pooling on the last hidden state. ~1.0 => reflection equivariant.
   An "interior" version trims `--edge` tokens at both ends to avoid FFT
   boundary artifacts.

2. pooled_reflection_cos  : cos( f(x) , f(reverse(x)) )
   f = L2-normalised mean pool. ~1.0 => pooled embedding cannot tell x from its
   reversal (the orientation-blind regime).

Anisotropy control
------------------
random_anchor_cos : cos( f(x_i) , f(x_j) ) for distinct random sequences.
If this is already ~0.98 (collapsed cone), the reflection numbers must be read
*relative to it* -- a high pooled_reflection_cos only means something when the
random anchor is low.

Usage
-----
    python src_hyena/diagnose_reflection_equivariance.py \
        --models \
            "H0:LongSafari/hyenadna-small-32k-seqlen-hf:plain" \
            "H1_tied:./hyena_bidir_h1:plain" \
            "H1_gated:./hyena_gated_bidir_h1:gated" \
            "H1_bidir:./hyena_honest_bidir_h1:bidir" \
        --num-seq 256 --seq-len 1024

`branch` selects which loader/patch to apply: plain (src_hyena), gated
(src_hyena_gated), bidir (src_hyena_bidir).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)

BRANCH_DIRS = {
    "plain": os.path.join(ROOT, "src_hyena"),
    "gated": os.path.join(ROOT, "src_hyena_gated"),
    "bidir": os.path.join(ROOT, "src_hyena_bidir"),
}


def load_branch_common(branch: str):
    """Import the branch-specific common.py in isolation.

    Every branch ships its own top-level ``hyena_bidirectional`` module, so we
    purge any cached copy and put the branch dir first on sys.path before the
    branch's common.py runs its ``from hyena_bidirectional import ...`` line.
    The resolved binding is captured at module-exec time, so later calls to
    ``load_hyena_backbone`` keep using the correct patch.
    """
    if branch not in BRANCH_DIRS:
        raise ValueError(f"Unknown branch {branch!r}; expected one of {list(BRANCH_DIRS)}.")
    branch_dir = BRANCH_DIRS[branch]
    common_path = os.path.join(branch_dir, "common.py")
    if not os.path.isfile(common_path):
        raise FileNotFoundError(common_path)

    for cached in ("hyena_bidirectional", "_old_hyena_common"):
        sys.modules.pop(cached, None)
    sys.path.insert(0, branch_dir)
    try:
        spec = importlib.util.spec_from_file_location(f"branch_common_{branch}", common_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(branch_dir)
        except ValueError:
            pass
    return module


def parse_spec(spec: str):
    name, path, branch = spec.rsplit(":", 2)
    return name, path, branch


def to_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def random_sequences(n: int, length: int, rng: np.random.Generator) -> list[str]:
    bases = np.array(list("ACGT"))
    idx = rng.integers(0, 4, size=(n, length))
    return ["".join(bases[row]) for row in idx]


def load_fasta_windows(path: str, n: int, length: int, rng: np.random.Generator) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    chunks.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
    if cur:
        chunks.append("".join(cur).upper())
    full = "".join(chunks)
    valid_max = len(full) - length
    if valid_max <= 0:
        raise ValueError("FASTA too short for the requested --seq-len.")
    seqs: list[str] = []
    tries = 0
    while len(seqs) < n and tries < n * 50:
        tries += 1
        start = int(rng.integers(0, valid_max))
        window = full[start : start + length]
        if set(window) <= set("ACGT"):
            seqs.append(window)
    if len(seqs) < n:
        raise ValueError(f"Only found {len(seqs)} clean windows; reduce --num-seq or --seq-len.")
    return seqs


@torch.inference_mode()
def hidden_states(model, common, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = common.extract_hidden_states(out)
    return hidden.float()


def evaluate_model(name, path, branch, sequences, args):
    common = load_branch_common(branch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_dtype(args.dtype)

    tokenizer = common.load_hyena_tokenizer(path)
    model, load_path = common.load_hyena_backbone(path, device=device, dtype=dtype)
    model.eval()

    impl = getattr(getattr(model, "config", None), "hyena_bidirectional_patch_impl", "none")
    print(f"\n-> {name}  (branch={branch}, impl={impl}, load={load_path})")

    tok_eq_full, tok_eq_inner, pooled_inv = [], [], []
    pooled_vecs = []

    for seq in sequences:
        enc = tokenizer(
            [seq],
            truncation=True,
            max_length=args.seq_len,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.size(1) < 2 * args.edge + 4:
            continue
        attn = torch.ones_like(input_ids)

        rev_ids = torch.flip(input_ids, dims=[1])

        h = hidden_states(model, common, input_ids, attn)        # [1, L, D]
        h_rev = hidden_states(model, common, rev_ids, attn)      # [1, L, D]

        h_flip = torch.flip(h, dims=[1])                         # reverse(H(x))
        cos_t = F.cosine_similarity(h_flip, h_rev, dim=-1).squeeze(0)  # [L]
        tok_eq_full.append(float(cos_t.mean()))
        if cos_t.numel() > 2 * args.edge:
            tok_eq_inner.append(float(cos_t[args.edge : cos_t.numel() - args.edge].mean()))

        f = F.normalize(h.mean(dim=1), dim=-1)                   # f(x)
        f_rev = F.normalize(h_rev.mean(dim=1), dim=-1)           # f(reverse(x))
        pooled_inv.append(float((f * f_rev).sum()))
        pooled_vecs.append(f.squeeze(0).cpu())

    # anisotropy control: cosine between distinct sequence embeddings
    anchors = []
    for i in range(1, len(pooled_vecs)):
        anchors.append(float((pooled_vecs[i] * pooled_vecs[i - 1]).sum()))

    def stats(v):
        a = np.asarray(v, dtype=np.float64)
        return a.mean(), np.median(a), np.percentile(a, 5), np.percentile(a, 95)

    te_m, te_med, te_p05, te_p95 = stats(tok_eq_full)
    ti_m, ti_med, ti_p05, ti_p95 = stats(tok_eq_inner) if tok_eq_inner else (te_m, te_med, te_p05, te_p95)
    pi_m, pi_med, pi_p05, pi_p95 = stats(pooled_inv)
    an_m, an_med, an_p05, an_p95 = stats(anchors) if anchors else (float("nan"),) * 4

    print(f"   token equivariance (full)  : mean={te_m:.4f} median={te_med:.4f} p05={te_p05:.4f}")
    print(f"   token equivariance (inner) : mean={ti_m:.4f} median={ti_med:.4f} p05={ti_p05:.4f}")
    print(f"   pooled reflection cos      : mean={pi_m:.4f} median={pi_med:.4f} p05={pi_p05:.4f}")
    print(f"   random anchor cos (aniso.) : mean={an_m:.4f} median={an_med:.4f} p05={an_p05:.4f}")
    margin = pi_m - an_m
    print(f"   reflection margin over aniso: {margin:+.4f}  "
          f"(high+ => orientation-blind; ~0 or negative => orientation preserved)")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "branch": branch,
        "impl": impl,
        "token_equivariance_full_mean": te_m,
        "token_equivariance_inner_mean": ti_m,
        "pooled_reflection_cos_mean": pi_m,
        "random_anchor_cos_mean": an_m,
        "reflection_margin_over_anisotropy": margin,
    }


def main():
    p = argparse.ArgumentParser(description="HyenaDNA reflection-equivariance diagnostic")
    p.add_argument("--models", nargs="+", required=True, metavar="name:path:branch")
    p.add_argument("--num-seq", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--edge", type=int, default=16, help="tokens trimmed each side for the interior metric")
    p.add_argument("--fasta", default=None, help="optional hg38 FASTA to sample real windows instead of random")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="optional JSON path for the summary table")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.fasta:
        sequences = load_fasta_windows(args.fasta, args.num_seq, args.seq_len, rng)
        source = f"fasta:{args.fasta}"
    else:
        sequences = random_sequences(args.num_seq, args.seq_len, rng)
        source = "random-ACGT"

    print("Reflection-equivariance diagnostic")
    print(f"  Sequences : {len(sequences)} ({source})")
    print(f"  Seq len   : {args.seq_len}  |  dtype: {args.dtype}  |  edge trim: {args.edge}")
    print("  Note      : 'reverse' = positional reversal (no complement).")

    rows = []
    for spec in args.models:
        name, path, branch = parse_spec(spec)
        rows.append(evaluate_model(name, path, branch, sequences, args))

    print("\n=== summary ===")
    print(f"{'model':<14}{'tok_eq_inner':>14}{'pooled_refl':>14}{'anchor':>10}{'margin':>10}")
    for r in rows:
        print(f"{r['name']:<14}{r['token_equivariance_inner_mean']:>14.4f}"
              f"{r['pooled_reflection_cos_mean']:>14.4f}{r['random_anchor_cos_mean']:>10.4f}"
              f"{r['reflection_margin_over_anisotropy']:>+10.4f}")

    if args.output:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"source": source, "seq_len": args.seq_len, "rows": rows}, handle, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
