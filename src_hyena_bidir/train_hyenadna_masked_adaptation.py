"""
Full Stage-2 MNTP entry point for the honest-bidirectional HyenaDNA branch.

This wrapper intentionally reuses the original training implementation while
keeping imports local to `src_hyena_bidir`, so `common.py` and
`hyena_bidirectional.py` resolve to the honest two-branch patch.
"""

from __future__ import annotations

import importlib.util
import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
OLD_HYENA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src_hyena")
if OLD_HYENA_DIR not in sys.path:
    sys.path.append(OLD_HYENA_DIR)

_OLD_TRAIN_PATH = os.path.join(OLD_HYENA_DIR, "train_hyenadna_masked_adaptation.py")
_spec = importlib.util.spec_from_file_location("_old_hyena_train_masked_adaptation", _OLD_TRAIN_PATH)
_old_train = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_old_train)
main = _old_train.main


def _enforce_full_only() -> None:
    if "--train-mode" in sys.argv:
        idx = sys.argv.index("--train-mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if mode != "full":
            raise SystemExit(
                "src_hyena_bidir only supports full Stage-2 MNTP. "
                "LoRA would freeze the reverse branch and direction gate."
            )
    else:
        sys.argv.extend(["--train-mode", "full"])

    if "--save-total-limit" not in sys.argv:
        sys.argv.extend(["--save-total-limit", "0"])


_enforce_full_only()


if __name__ == "__main__":
    main()
