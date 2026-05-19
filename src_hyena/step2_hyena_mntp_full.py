"""
HyenaDNA Step 2 full-fine-tuning entry point.

This wrapper preserves the original H2 recipe: span-masked MNTP with full-model
adaptation. Use step2_hyena_mntp_lora.py for the parameter-efficient diagnostic.
"""

from __future__ import annotations

import sys

from train_hyenadna_masked_adaptation import main


def _inject_full_defaults() -> None:
    if "--train-mode" not in sys.argv:
        sys.argv.extend(["--train-mode", "full"])
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "./hyena_h2_masked_adapted"])
    if "--run-name" not in sys.argv:
        sys.argv.extend(["--run-name", "hyena_step2_span_masked_mntp_full"])


_inject_full_defaults()


if __name__ == "__main__":
    main()
