"""
Alias entry point for HyenaDNA Step 5 LoRA fine-tuning.

The implementation lives in step5_hyena_finetune_eval.py; this filename keeps
the LoRA/full distinction explicit next to step5_hyena_full_finetune_eval.py.
"""

from __future__ import annotations

from step5_hyena_finetune_eval import base


if __name__ == "__main__":
    base.main()
