"""
HyenaDNA Step 3 LoRA entry point for all contrastive modes.

Use ``--mode`` to select dropout, reverse-complement, crop or local-shift
SimCSE. This wrapper keeps the main Step 3 implementation in
step3_hyena_contrastive_lora.py and only changes the parameter-efficient
LoRA recipe defaults.
"""

from __future__ import annotations

import sys

from step3_hyena_contrastive_lora import main


LORA_DEFAULTS = {
    "--train-mode": "lora",
    "--lr": "1e-4",
    "--lora-r": "16",
    "--lora-alpha": "32",
    "--lora-dropout": "0.1",
    "--lora-target-modules": "in_proj,out_proj,fc1,fc2",
}


def _has_option(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


def _inject_lora_defaults() -> None:
    for flag, value in LORA_DEFAULTS.items():
        if not _has_option(flag):
            sys.argv.extend([flag, value])


_inject_lora_defaults()


if __name__ == "__main__":
    main()
