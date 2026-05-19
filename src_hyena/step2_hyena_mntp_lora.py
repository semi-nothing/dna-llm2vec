"""
HyenaDNA Step 2 LoRA entry point.

This wrapper runs span-masked MNTP with parameter-efficient LoRA adaptation
instead of the original full-model HyenaDNA Step 2 update. By default it targets
the conservative projection/MLP modules:
  in_proj,out_proj,fc1,fc2

The underlying training implementation is shared with
train_hyenadna_masked_adaptation.py; this file only sets LoRA-specific defaults
so LoRA and full fine-tuning runs are easy to distinguish.
"""

from __future__ import annotations

import sys

from train_hyenadna_masked_adaptation import main


def _inject_lora_defaults() -> None:
    if "--train-mode" not in sys.argv:
        sys.argv.extend(["--train-mode", "lora"])
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "./hyena_h2_mntp_lora"])
    if "--run-name" not in sys.argv:
        sys.argv.extend(["--run-name", "hyena_step2_mntp_lora"])


_inject_lora_defaults()


if __name__ == "__main__":
    main()
