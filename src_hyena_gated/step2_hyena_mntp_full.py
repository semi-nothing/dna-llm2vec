"""
HyenaDNA gated-tied Step 2 full-model MNTP entry point.
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
OLD_HYENA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src_hyena")
if OLD_HYENA_DIR not in sys.path:
    sys.path.append(OLD_HYENA_DIR)

from train_hyenadna_masked_adaptation import main  # noqa: E402


def _inject_full_defaults() -> None:
    if "--train-mode" in sys.argv:
        idx = sys.argv.index("--train-mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if mode != "full":
            raise SystemExit(
                "src_hyena_gated only supports full Stage-2 MNTP. "
                "LoRA would freeze the learned direction gates."
            )
    if "--train-mode" not in sys.argv:
        sys.argv.extend(["--train-mode", "full"])
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "./hyena_gated_bidir_h1"])
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "./hyena_gated_bidir_h2_mntp_full"])
    if "--run-name" not in sys.argv:
        sys.argv.extend(["--run-name", "hyena_gated_bidir_step2_mntp_full"])


_inject_full_defaults()


if __name__ == "__main__":
    main()
