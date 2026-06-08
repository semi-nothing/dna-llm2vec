"""
Full Step-3 contrastive entry point for the gated-tied HyenaDNA branch.

This wrapper reuses the original Step-3 implementation while resolving
`common.py` and `hyena_bidirectional.py` from `src_hyena_gated`.
"""

from __future__ import annotations

import importlib.util
import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
OLD_HYENA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src_hyena")
if OLD_HYENA_DIR not in sys.path:
    sys.path.append(OLD_HYENA_DIR)

_OLD_STEP3_PATH = os.path.join(OLD_HYENA_DIR, "step3_hyena_contrastive_lora.py")
_spec = importlib.util.spec_from_file_location("_old_hyena_step3_contrastive", _OLD_STEP3_PATH)
_old_step3 = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _old_step3
_spec.loader.exec_module(_old_step3)
main = _old_step3.main


def _enforce_full_only() -> None:
    if "--train-mode" in sys.argv:
        idx = sys.argv.index("--train-mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if mode != "full":
            raise SystemExit(
                "src_hyena_gated Step 3 only supports full contrastive training. "
                "LoRA would freeze the learned direction gates."
            )
    else:
        sys.argv.extend(["--train-mode", "full"])

    if "--save-total-limit" not in sys.argv:
        sys.argv.extend(["--save-total-limit", "0"])


_enforce_full_only()


if __name__ == "__main__":
    main()
