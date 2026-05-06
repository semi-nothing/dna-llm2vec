# DNA-LLM2Vec

DNA-LLM2Vec converts pretrained decoder-style genomic foundation models into
bidirectional sequence encoders. The repository contains the staged training
pipeline used in the paper:

- **Stage 1:** open bidirectional context.
- **Stage 2:** masked next-token prediction (MNTP) adaptation.
- **Stage 3:** SimCSE-style contrastive adaptation with genomic positive pairs.
- **Step 4/5:** downstream evaluation by frozen linear probe and task-specific
  fine-tuning.

The DNAGPT branch produces variants M0--M6:

| Variant | Description |
| --- | --- |
| M0 | Original causal DNAGPT checkpoint |
| M1 | Bidirectional attention patch |
| M2 | MNTP-adapted bidirectional model |
| M3 | Dropout SimCSE |
| M4 | Reverse-complement SimCSE |
| M5 | Crop SimCSE |
| M6 | Local-shift SimCSE |

## Setup

This project uses `uv` for environment management.

```bash
uv sync
```

The examples below assume the hg38 FASTA is available at:

```text
./data/hg38.fa
```

All stochastic training stages should be repeated with seeds `42`, `43`, and
`44` for the reported multi-seed results.

## DNAGPT Training Pipeline

### Stage 1: Open Bidirectional Attention

This stage converts the public causal DNAGPT checkpoint into a bidirectional
checkpoint without updating any parameters.

```bash
uv run python src/step1_bidirectional.py \
  --model dnagpt/human_gpt2-v1 \
  --output ./bidir_dnagpt
```

Output:

```text
./bidir_dnagpt
```

### Stage 2: MNTP Adaptation

This stage trains a LoRA-adapted bidirectional DNAGPT model with the mask-only
embedding update policy. Only the added `[MASK]` embedding row is updated in the
token embedding table; the original DNAGPT BPE embeddings remain frozen.

Example for seed 42:

```bash
uv run python src/step2_mntp_lora_mask_only.py \
  --model ./bidir_dnagpt \
  --fasta ./data/hg38.fa \
  --output ./mntp_dnagpt_lora_fn_mask_only_s42_r1 \
  --filter-n \
  --max-length 1024 \
  --epochs 1 \
  --max-steps -1 \
  --batch-size 24 \
  --grad-accum 2 \
  --lr 1e-4 \
  --warmup-steps 500 \
  --mlm-probability 0.15 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 42 \
  --repeat-index 1 \
  --run-name step2_mntp_lora_fn_mask_only_s42_r1
```

Repeat with the corresponding output names, seeds, and repeat indices for
additional runs, for example `s43_r2` and `s44_r3`.

### Stage 3: Contrastive Adaptation

Stage 3 starts from an M2 checkpoint and trains one of four contrastive variants.
All examples below use seed 43 / repeat 2; adjust `--model`, `--output`,
`--seed`, `--repeat-index`, and `--run-name` for other seeds.

#### M5: Crop SimCSE

Two overlapping 4096 bp crops are sampled from the same genomic window. This is
the strongest frozen-representation variant on long-range EPI in the paper.

```bash
uv run python src/step3_contrastive_lora.py \
  --model ./mntp_dnagpt_lora_fn_mask_only_s43_r2 \
  --fasta ./data/hg38.fa \
  --output ./contrastive_dnagpt_crop_lora_fn_mask_only_s43_r2 \
  --mode crop \
  --filter-n \
  --max-length 1024 \
  --max-steps -1 \
  --epochs 1 \
  --chunk-size 4096 \
  --overlap-ratio 0.5 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --grad-ckpt \
  --seed 43 \
  --repeat-index 2 \
  --run-name step3_crop_lora_fn_mask_only_s43_r2
```

#### M3: Dropout SimCSE

The same sequence is encoded twice with independent dropout masks.

```bash
uv run python src/step3_contrastive_lora.py \
  --model ./mntp_dnagpt_lora_fn_mask_only_s43_r2 \
  --fasta ./data/hg38.fa \
  --output ./contrastive_dnagpt_dropout_lora_fn_mask_only_s43_r2 \
  --mode dropout \
  --filter-n \
  --max-length 1024 \
  --max-steps 3000 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 1e-4 \
  --dropout 0.3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 43 \
  --grad-ckpt \
  --repeat-index 2 \
  --run-name step3_dropout_lora_fn_mask_only_s43_r2
```

#### M6: Local-Shift SimCSE

The positive view is a small local shift of the anchor crop from the same
genomic window.

```bash
uv run python src/step3_contrastive_lora.py \
  --model ./mntp_dnagpt_lora_fn_mask_only_s43_r2 \
  --fasta ./data/hg38.fa \
  --output ./contrastive_dnagpt_local_shift_lora_fn_mask_only_s43_r2 \
  --mode local_shift \
  --filter-n \
  --max-length 1024 \
  --max-steps -1 \
  --epochs 1 \
  --chunk-size 4096 \
  --local-shift-ratio 0.1 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --grad-ckpt \
  --seed 43 \
  --repeat-index 2 \
  --run-name step3_local_shift_lora_fn_mask_only_s43_r2
```

#### M4: Reverse-Complement SimCSE

The positive view is the reverse complement of the input sequence.

```bash
uv run python src/step3_contrastive_lora.py \
  --model ./mntp_dnagpt_lora_fn_mask_only_s43_r2 \
  --fasta ./data/hg38.fa \
  --output ./contrastive_dnagpt_revcomp_lora_fn_mask_only_s43_r2 \
  --mode revcomp \
  --filter-n \
  --max-length 1024 \
  --max-steps -1 \
  --epochs 1 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --grad-ckpt \
  --seed 43 \
  --repeat-index 2 \
  --run-name step3_revcomp_lora_fn_mask_only_s43_r2
```

## DNAGPT Evaluation Pipeline

### Step 4: Frozen Linear Probe

Step 4 evaluates frozen representations with mean pooling and a lightweight
linear probe. The example below evaluates the causal baseline (M0) and the
bidirectional patch-only model (M1) on all configured benchmark suites, including
GUE+ EPI with junction cropping.

```bash
uv run python src/step4_evaluate.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M1:./bidir_dnagpt:bidir" \
  --gue-plus-dir ./data/GUE_plus \
  --epi-crop-bp 4096 \
  --epi-crop-mode junction \
  --filter-n \
  --batch-size 128 \
  --max-length 1024 \
  --pooling mean \
  --seed 42 \
  --repeat-index 1 \
  --gb-root /data/home/bty252/.genomic_benchmarks \
  --benchmark-cache-dir ./cache/benchmarks \
  --run-name step4_all_mean_junction_M0_M1_s42_r1 \
  --output ./eval_results/step4_all_mean_junction_M0_M1_s42_r1.json
```

To evaluate M2--M6, add the corresponding model specifications to `--models`,
using the format:

```text
"NAME:PATH:MODE"
```

where `MODE` is `causal` for M0 and `bidir` for bidirectional/adapted
checkpoints.

### Step 5: LoRA Fine-Tuning

Step 5 LoRA fine-tunes a task-specific LoRA head for each downstream task while
keeping the backbone parameter-efficiently adapted. The example below evaluates
GUE for all DNAGPT variants under both maskfix and mask-only checkpoints.

```bash
uv run python src/step5_finetune_eval.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M1:./bidir_dnagpt:bidir" \
    "M2_maskfix_r3:./mntp_dnagpt_lora_fn_maskfix_s44_r3:bidir" \
    "M2_mask_only_r3:./mntp_dnagpt_lora_fn_mask_only_s44_r3:bidir" \
    "M3_maskfix_r3:./contrastive_dnagpt_dropout_lora_fn_maskfix_s44_r3:bidir" \
    "M3_mask_only_r3:./contrastive_dnagpt_dropout_lora_fn_mask_only_s44_r3:bidir" \
    "M4_maskfix_r3:./contrastive_dnagpt_revcomp_lora_fn_maskfix_s44_r3:bidir" \
    "M4_mask_only_r3:./contrastive_dnagpt_revcomp_lora_fn_mask_only_s44_r3:bidir" \
    "M5_maskfix_r3:./contrastive_dnagpt_crop_lora_fn_maskfix_s44_r3:bidir" \
    "M5_mask_only_r3:./contrastive_dnagpt_crop_lora_fn_mask_only_s44_r3:bidir" \
    "M6_maskfix_r3:./contrastive_dnagpt_local_shift_lora_fn_maskfix_s44_r3:bidir" \
    "M6_mask_only_r3:./contrastive_dnagpt_local_shift_lora_fn_mask_only_s44_r3:bidir" \
  --gue-only \
  --filter-n \
  --epochs 10 \
  --batch-size 96 \
  --grad-accum 1 \
  --max-length 1024 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --grad-clip 1.0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --no-wandb \
  --seed 44 \
  --repeat-index 3 \
  --run-name step5lora_gue_M0_M1_M2_M3_M4_M5_M6_r3_maskfix_maskonly \
  --output ./eval_results/step5lora_gue_M0_M1_M2_M3_M4_M5_M6_r3_maskfix_maskonly.json
```

### Step 5: Full Fine-Tuning

Full fine-tuning updates all model parameters for each downstream task. The
example below runs full fine-tuning on GUE+ EPI with the same 4096 bp junction
crop used in the frozen-representation analyses.

```bash
uv run python src/step5_full_finetune_eval.py \
  --models \
    "M0:dnagpt/human_gpt2-v1:causal" \
    "M1:./bidir_dnagpt:bidir" \
    "M2_maskfix_r2:./mntp_dnagpt_lora_fn_maskfix_s43_r2:bidir" \
    "M2_mask_only_r2:./mntp_dnagpt_lora_fn_mask_only_s43_r2:bidir" \
    "M3_maskfix_r2:./contrastive_dnagpt_dropout_lora_fn_maskfix_s43_r2:bidir" \
    "M3_mask_only_r2:./contrastive_dnagpt_dropout_lora_fn_mask_only_s43_r2:bidir" \
    "M4_maskfix_r2:./contrastive_dnagpt_revcomp_lora_fn_maskfix_s43_r2:bidir" \
    "M4_mask_only_r2:./contrastive_dnagpt_revcomp_lora_fn_mask_only_s43_r2:bidir" \
    "M5_maskfix_r2:./contrastive_dnagpt_crop_lora_fn_maskfix_s43_r2:bidir" \
    "M5_mask_only_r2:./contrastive_dnagpt_crop_lora_fn_mask_only_s43_r2:bidir" \
    "M6_maskfix_r2:./contrastive_dnagpt_local_shift_lora_fn_maskfix_s43_r2:bidir" \
    "M6_mask_only_r2:./contrastive_dnagpt_local_shift_lora_fn_mask_only_s43_r2:bidir" \
  --gue-plus-only \
  --gue-plus-dir ./data/GUE_plus \
  --epi-crop-bp 4096 \
  --epi-crop-mode junction \
  --max-length 1024 \
  --epochs 20 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 3e-5 \
  --warmup-steps 50 \
  --grad-ckpt \
  --monitor mcc \
  --patience 0 \
  --min-delta 0.0 \
  --no-wandb \
  --seed 43 \
  --repeat-index 2 \
  --run-name step5full_epi_junction_M0_M1_M2_M3_M4_M5_M6_maskfix_maskonly_r2 \
  --output ./eval_results/step5full_epi_junction_M0_M1_M2_M3_M4_M5_M6_maskfix_maskonly_r2.json
```

## HyenaDNA Branch

The HyenaDNA branch follows the same conceptual stages (H0--H6), but uses
Hyena-specific scripts in `src_hyena/`:

- `src_hyena/step1_hyena.py`
- `src_hyena/train_hyenadna_masked_adaptation.py`
- `src_hyena/step3_hyena_contrastive_lora.py`

HyenaDNA commands will be added here after the final training commands are
fixed.

## Notes

- Use `--filter-n` to exclude windows containing ambiguous bases.
- DNAGPT uses BPE tokenisation with `--max-length 1024`.
- For DNAGPT crop/local-shift contrastive modes, `--chunk-size 4096` is a raw
  base-pair length used before tokenisation.
- Reported stochastic stages should be repeated with seeds `42`, `43`, and
  `44`.
