#!/usr/bin/env python
"""
Honest-bidirectional HyenaDNA VEP/SNP distance-bucket probe.

This branch entry point reuses `src/evaluate_vep_distance_probe.py`, while
resolving `step4_hyena_evaluate.py` from `src_hyena_bidir`. Use `:hyena` for
HyenaDNA checkpoints so the honest-bidir wrapper is applied before embedding
extraction.

Example:
  uv run python src_hyena_bidir/evaluate_vep_distance_probe.py \
    --models "H0:LongSafari/hyenadna-small-32k-seqlen-hf:hyena" \
             "H1_bidir:./hyena_bidir_h1:hyena" \
    --csv ./data/vep_windows.csv \
    --feature diff \
    --probes linear rbf_svm \
    --train-per-bucket 5000 \
    --repeats 5 \
    --output ./eval_results/vep_hyena_bidir_probe.json
"""

from __future__ import annotations

import importlib.util
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
LEGACY_HYENA_DIR = os.path.join(ROOT, "src_hyena")
CORE_PATH = os.path.join(SRC_DIR, "evaluate_vep_distance_probe.py")


def _prioritize_branch_imports() -> None:
    for path in (CURRENT_DIR, SRC_DIR, ROOT, LEGACY_HYENA_DIR):
        if path in sys.path:
            sys.path.remove(path)
    for path in reversed((CURRENT_DIR, SRC_DIR, ROOT, LEGACY_HYENA_DIR)):
        sys.path.insert(0, path)


def main() -> None:
    _prioritize_branch_imports()
    spec = importlib.util.spec_from_file_location("_vep_distance_probe_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load VEP probe core from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
