#!/usr/bin/env python
"""
DNAGPT M0-M6 preset for the frozen VEP/SNP distance-bucket probe.

DNAGPT uses 1024 tokens, which is roughly a 4096 bp DNA window for the
human_gpt2-v1 tokenizer. This entry point reuses
`src/evaluate_vep_distance_probe.py` and fills in the standard M0-M6 model
list plus DNAGPT-friendly defaults.

Default preset is the mask-only LoRA r1 series used in the current paper runs.
Use `--preset full` for the older full-checkpoint names, or pass `--models`
explicitly to override everything.

Example:
  uv run python src/evaluate_vep_m0_m6.py \
    --csv ./data/vep_windows.csv \
    --feature caduceus_concat \
    --probes rbf_svm \
    --train-per-bucket 5000 \
    --repeats 5
"""

from __future__ import annotations

import importlib.util
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
CORE_PATH = os.path.join(SRC_DIR, "evaluate_vep_distance_probe.py")

MASK_ONLY_R1_MODELS = [
    "M0:dnagpt/human_gpt2-v1:causal",
    "M1:./bidir_dnagpt:bidir",
    "M2:./mntp_dnagpt_lora_fn_mask_only_s42_r1:bidir",
    "M3:./contrastive_dnagpt_dropout_lora_fn_mask_only_s42_r1:bidir",
    "M4:./contrastive_dnagpt_revcomp_lora_fn_mask_only_s42_r1:bidir",
    "M5:./contrastive_dnagpt_crop_lora_fn_mask_only_s42_r1:bidir",
    "M6:./contrastive_dnagpt_localshift_lora_fn_mask_only_s42_r1:bidir",
]

FULL_MODELS = [
    "M0:dnagpt/human_gpt2-v1:causal",
    "M1:./bidir_dnagpt:bidir",
    "M2:./mntp_dnagpt:bidir",
    "M3:./contrastive_dnagpt_dropout:bidir",
    "M4:./contrastive_dnagpt_revcomp:bidir",
    "M5:./contrastive_dnagpt_crop:bidir",
    "M6:./contrastive_dnagpt_localshift:bidir",
]


def _pop_preset(argv: list[str]) -> tuple[list[str], str]:
    out = []
    preset = "mask-only-r1"
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--preset":
            if i + 1 >= len(argv):
                raise SystemExit("--preset requires one of: mask-only-r1, full")
            preset = argv[i + 1]
            i += 2
            continue
        if item.startswith("--preset="):
            preset = item.split("=", 1)[1]
            i += 1
            continue
        out.append(item)
        i += 1

    if preset not in {"mask-only-r1", "full"}:
        raise SystemExit(f"Unsupported --preset {preset!r}; choose mask-only-r1 or full.")
    return out, preset


def _insert_defaults(argv: list[str], preset: str) -> list[str]:
    models = MASK_ONLY_R1_MODELS if preset == "mask-only-r1" else FULL_MODELS
    out = list(argv)

    if "--models" not in out:
        out.extend(["--models", *models])
    if "--max-length" not in out:
        out.extend(["--max-length", "1024"])
    if "--output" not in out:
        suffix = "mask_only_r1" if preset == "mask-only-r1" else "full"
        out.extend(["--output", f"./eval_results/vep_dnagpt_m0_m6_{suffix}.json"])
    if "--baseline-model" not in out:
        out.extend(["--baseline-model", "M0"])

    return out


def main() -> None:
    args, preset = _pop_preset(sys.argv[1:])
    sys.argv = [sys.argv[0], *_insert_defaults(args, preset)]

    spec = importlib.util.spec_from_file_location("_vep_distance_probe_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load VEP probe core from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
