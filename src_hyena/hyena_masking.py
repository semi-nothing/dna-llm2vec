"""
Masking utilities for HyenaDNA Step 2 adaptation.

Supports single-base masking and nucleotide span masking at single-base
resolution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class SpanMaskingConfig:
    mask_probability: float = 0.15
    masking_mode: str = "span"   # "single" | "span"
    span_min_length: int = 3
    span_max_length: int = 20
    replacement_policy: str = "all_mask"  # "all_mask" | "bert" | "mask_random"


def _select_single_mask_positions(valid_positions: list[int], mask_probability: float) -> set[int]:
    return {pos for pos in valid_positions if random.random() < mask_probability}


def _select_span_mask_positions(
    valid_positions: list[int],
    mask_probability: float,
    span_min_length: int,
    span_max_length: int,
) -> set[int]:
    if not valid_positions:
        return set()

    valid_set = set(valid_positions)
    target = max(1, int(round(len(valid_positions) * mask_probability)))
    masked: set[int] = set()
    attempts = 0
    max_attempts = max(16, target * 8)

    while len(masked) < target and attempts < max_attempts:
        attempts += 1
        start = random.choice(valid_positions)
        span_len = random.randint(span_min_length, span_max_length)
        for pos in range(start, start + span_len):
            if pos not in valid_set:
                break
            masked.add(pos)
            if len(masked) >= target:
                break

    return masked


class HyenaDNASpanMaskingCollator:
    def __init__(self, tokenizer, config: SpanMaskingConfig):
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must define a dedicated mask token before masking.")
        if config.replacement_policy not in {"all_mask", "bert", "mask_random"}:
            raise ValueError(f"Unsupported replacement_policy={config.replacement_policy!r}")
        self.tokenizer = tokenizer
        self.config = config
        base_ids = [
            tokenizer.convert_tokens_to_ids(base)
            for base in ("A", "C", "G", "T")
        ]
        self.random_token_ids = torch.tensor(
            [int(token_id) for token_id in base_ids if token_id is not None],
            dtype=torch.long,
        )
        if len(self.random_token_ids) != 4:
            raise ValueError("HyenaDNA random replacement requires single-token A/C/G/T ids.")

    def _apply_replacement_policy(self, input_ids: torch.Tensor, idx: torch.Tensor) -> None:
        policy = self.config.replacement_policy
        if policy == "all_mask":
            input_ids[idx] = self.tokenizer.mask_token_id
            return

        probs = torch.rand(idx.numel(), device=input_ids.device)
        mask_positions = probs < (0.8 if policy == "bert" else 0.9)
        if mask_positions.any():
            input_ids[idx[mask_positions]] = self.tokenizer.mask_token_id

        random_positions = (probs >= 0.8) & (probs < 0.9) if policy == "bert" else probs >= 0.9
        if random_positions.any():
            choices = self.random_token_ids.to(device=input_ids.device)
            rand_idx = torch.randint(0, choices.numel(), (int(random_positions.sum()),), device=input_ids.device)
            input_ids[idx[random_positions]] = choices[rand_idx]

    def _mask_one(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor):
        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        valid_positions = [
            idx
            for idx in range(input_ids.shape[0])
            if attention_mask[idx].item() == 1 and special_tokens_mask[idx].item() == 0
        ]

        if self.config.masking_mode == "single":
            masked_positions = _select_single_mask_positions(valid_positions, self.config.mask_probability)
        elif self.config.masking_mode == "span":
            masked_positions = _select_span_mask_positions(
                valid_positions=valid_positions,
                mask_probability=self.config.mask_probability,
                span_min_length=self.config.span_min_length,
                span_max_length=self.config.span_max_length,
            )
        else:
            raise ValueError(f"Unsupported masking_mode={self.config.masking_mode!r}")

        if not masked_positions and valid_positions:
            masked_positions = {random.choice(valid_positions)}

        if masked_positions:
            idx = torch.tensor(sorted(masked_positions), dtype=torch.long)
            labels[idx] = input_ids[idx]
            self._apply_replacement_policy(input_ids, idx)

        return input_ids, labels

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.as_tensor(f["input_ids"]) for f in features])

        if "attention_mask" in features[0]:
            attention_mask = torch.stack([torch.as_tensor(f["attention_mask"]) for f in features])
        else:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = (input_ids != pad_id).long()

        if "special_tokens_mask" in features[0]:
            special_tokens_mask = torch.stack([torch.as_tensor(f["special_tokens_mask"]) for f in features])
        else:
            special_tokens_mask = torch.zeros_like(input_ids)

        masked_ids = []
        labels = []
        for ids_row, attn_row, special_row in zip(input_ids, attention_mask, special_tokens_mask):
            ids_out, labels_out = self._mask_one(ids_row, attn_row, special_row)
            masked_ids.append(ids_out)
            labels.append(labels_out)

        return {
            "input_ids": torch.stack(masked_ids),
            "attention_mask": attention_mask,
            "labels": torch.stack(labels),
        }
