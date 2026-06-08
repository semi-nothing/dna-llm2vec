"""
Step 1 for the gated-tied HyenaDNA branch.

H1 keeps shared forward/reverse taps but replaces fixed 0.5 averaging with
learnable per-channel gates.
"""

from __future__ import annotations

import importlib.util
import os
import sys

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


def _inject_gated_defaults() -> None:
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", "./hyena_gated_bidir_h1"])


_inject_gated_defaults()


if __name__ == "__main__":
    main()
