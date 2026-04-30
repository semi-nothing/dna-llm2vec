"""
Compatibility wrapper for the HyenaDNA Step 2 adaptation script.

HyenaDNA Step 2 defaults to full fine-tuning rather than LoRA, because the
backbone is much smaller than the DNAGPT branch.
"""

from train_hyenadna_masked_adaptation import main


if __name__ == "__main__":
    main()
