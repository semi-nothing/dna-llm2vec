"""
Step 1 for the honest-bidirectional HyenaDNA branch.

H0: pretrained causal HyenaDNA baseline
H1: pretrained weights + untied forward/reverse Hyena branches + learnable merge
"""

from __future__ import annotations

import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(__file__)
OLD_HYENA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src_hyena")
if OLD_HYENA_DIR not in sys.path:
    sys.path.append(OLD_HYENA_DIR)

_OLD_STEP1_PATH = os.path.join(OLD_HYENA_DIR, "step1_hyena.py")
_spec = importlib.util.spec_from_file_location("_old_hyena_step1", _OLD_STEP1_PATH)
_old_step1 = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _old_step1
_spec.loader.exec_module(_old_step1)
main = _old_step1.main


def _inject_honest_defaults() -> None:
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "./hyena_honest_bidir_h1"])


_inject_honest_defaults()


if __name__ == "__main__":
    main()
