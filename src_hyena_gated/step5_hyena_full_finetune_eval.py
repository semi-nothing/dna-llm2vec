"""
HyenaDNA Step 5 full fine-tuning wrapper for the gated-tied branch.

This reuses the shared HyenaDNA full fine-tuning implementation while resolving
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

# Preload branch-local modules before the shared implementation inserts
# src_hyena at the front of sys.path.
import hyena_bidirectional  # noqa: F401,E402
import common  # noqa: F401,E402

_OLD_STEP5_PATH = os.path.join(OLD_HYENA_DIR, "step5_hyena_full_finetune_eval.py")
_spec = importlib.util.spec_from_file_location("_old_hyena_step5_full_finetune_eval", _OLD_STEP5_PATH)
_old_step5 = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _old_step5
_spec.loader.exec_module(_old_step5)


if __name__ == "__main__":
    _old_step5.base.main()
