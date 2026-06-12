"""
DNA-LLM2Vec  |  Step 4: Evaluation — Genomics Benchmarks + NT + GUE Downstream Tasks
======================================================================================
Evaluates DNA embeddings using a linear probe on three benchmark suites:

  GB  — Genomics Benchmarks (Grevsova et al., 2023), 8 tasks.
        Core evaluation suite; DNABERT-2 literature has published numbers.
        Loaded via the genomic-benchmarks Python package.
        Conventional display metric: accuracy.

  NT  — Nucleotide Transformer downstream tasks (Dalla-Torre et al., 2023),
        18 tasks across promoter, enhancer, splice-site, and histone-mark
        classification.
        Shows generalisation beyond GB. Loaded from HuggingFace Hub.
        Conventional display metric: accuracy, while F1 and MCC are also computed.

  GUE — Genome Understanding Evaluation (Ji et al., 2021; used by DNABERT-2),
        18 tasks across promoter detection, TF binding, splicing, virus.
        Loaded from leannmlindsey/GUE on HuggingFace Hub.
        Primary metric: F1 (macro) + MCC, matching DNABERT-2 paper convention.
        Trained on train split, evaluated on dev split.

Running GB + NT + GUE is the default. Pass --gue-plus-dir to include the
six GUE+ EPI tasks too, or use --gue-plus-only to run only EPI.

Model spec format  →  name:path:mode
  name  — label in results table (e.g. M0, M2)
  path  — local directory or HuggingFace model ID
  mode  — "causal" (keep causal mask) | "bidir" (apply Step 1 patch) | "encoder" (bidirectional encoder, e.g. NT-500M/ESM)

Usage
-----
  # M0 vs M2 (all suites):
  uv run python src/step4_evaluate.py \\
      --models "M0:dnagpt/human_gpt2-v1:causal" \\
               "M2:./mntp_dnagpt:bidir" \\
      --output ./eval_results/m0_vs_m2.json

  # Full ablation M0–M4:
  uv run python src/step4_evaluate.py \\
      --models "M0:dnagpt/human_gpt2-v1:causal" \\
               "M1:./bidir_dnagpt:bidir" \\
               "M2:./mntp_dnagpt:bidir" \\
               "M3:./contrastive_dnagpt_dropout:bidir" \\
               "M4:./contrastive_dnagpt_revcomp:bidir" \\
      --output ./eval_results/full_ablation.json

  # GUE only (compare with DNABERT-2):
  uv run python src/step4_evaluate.py \\
      --models "M0:dnagpt/human_gpt2-v1:causal" \\
               "M4:./contrastive_dnagpt_revcomp:bidir" \\
      --gue-only --output ./eval_results/gue.json

  # GB only (quick sanity check):
  uv run python src/step4_evaluate.py \\
      --models "M2:./mntp_dnagpt:bidir" --gb-only \\
      --output ./eval_results/m2_gb.json
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from step1_bidirectional import patch_to_bidirectional


# ── Benchmark registries ──────────────────────────────────────────────────────

# (task_key, n_classes, display_name, source)
#   source = "gb"  → loaded via genomic-benchmarks package
#   source = "nt"  → loaded from InstaDeepAI/nucleotide_transformer_downstream_tasks
#   source = "gue" → loaded from leannmlindsey/GUE

GB_BENCHMARKS = [
    ("demo_coding_vs_intergenomic_seqs", 2, "Coding-vs-Intergenic", "gb"),
    ("demo_human_or_worm",               2, "Human-vs-Worm",        "gb"),
    ("human_enhancers_cohn",          2, "Enh-Cohn",     "gb"),
    ("human_enhancers_ensembl",        2, "Enh-Ensembl",  "gb"),
    ("human_ensembl_regulatory",       3, "Regulatory",   "gb"),
    ("human_nontata_promoters",        2, "Non-TATA",     "gb"),
    ("human_ocr_ensembl",              2, "OCR",          "gb"),
    ("dummy_mouse_enhancers_ensembl",  2, "Mouse-Enh",   "gb"),
]

NT_BENCHMARKS = [
    ("promoter_all",          2, "Promoter-All",        "nt"),
    ("promoter_tata",         2, "Promoter-TATA",       "nt"),
    ("promoter_no_tata",      2, "Promoter-noTATA",     "nt"),
    ("enhancers",             2, "Enhancers",           "nt"),
    ("enhancers_types",       3, "Enhancer-Types",      "nt"),
    ("splice_sites_all",      3, "Splice-All",          "nt"),
    ("splice_sites_acceptor", 2, "Splice-Acceptor",     "nt"),
    ("splice_sites_donor",    2, "Splice-Donor",        "nt"),
    ("H3",                    2, "H3",                  "nt"),
    ("H4",                    2, "H4",                  "nt"),
    ("H3K9ac",                2, "H3K9ac",              "nt"),
    ("H3K14ac",               2, "H3K14ac",             "nt"),
    ("H4ac",                  2, "H4ac",                "nt"),
    ("H3K4me1",               2, "H3K4me1",             "nt"),
    ("H3K4me2",               2, "H3K4me2",             "nt"),
    ("H3K4me3",               2, "H3K4me3",             "nt"),
    ("H3K36me3",              2, "H3K36me3",            "nt"),
    ("H3K79me3",              2, "H3K79me3",            "nt"),
]

# 18 GUE tasks — matches DNABERT-2 evaluation suite.
# Primary metrics: F1 (macro) and MCC (not accuracy).
# Trained on train split, evaluated on dev split.
GUE_BENCHMARKS = [
    # Promoter detection (core, ~70 bp)
    ("prom_core_all",        2, "Prom-Core-All",   "gue"),
    ("prom_core_tata",       2, "Prom-Core-TATA",  "gue"),
    ("prom_core_notata",     2, "Prom-Core-noTATA","gue"),
    # Promoter detection (300 bp)
    ("prom_300_all",         2, "Prom-300-All",    "gue"),
    ("prom_300_tata",        2, "Prom-300-TATA",   "gue"),
    ("prom_300_notata",      2, "Prom-300-noTATA", "gue"),
    # Transcription factor binding — human (5 cell-line splits)
    ("human_tf_0",           2, "TF-Human-0",      "gue"),
    ("human_tf_1",           2, "TF-Human-1",      "gue"),
    ("human_tf_2",           2, "TF-Human-2",      "gue"),
    ("human_tf_3",           2, "TF-Human-3",      "gue"),
    ("human_tf_4",           2, "TF-Human-4",      "gue"),
    # Transcription factor binding — mouse (5 splits)
    ("mouse_0",              2, "TF-Mouse-0",      "gue"),
    ("mouse_1",              2, "TF-Mouse-1",      "gue"),
    ("mouse_2",              2, "TF-Mouse-2",      "gue"),
    ("mouse_3",              2, "TF-Mouse-3",      "gue"),
    ("mouse_4",              2, "TF-Mouse-4",      "gue"),
    # Splice site prediction
    ("splice_reconstructed", 3, "Splice",          "gue"),
    # Virus COVID classification
    ("virus_covid",          2, "Virus-COVID",     "gue"),
]

ALL_BENCHMARKS = GB_BENCHMARKS + NT_BENCHMARKS + GUE_BENCHMARKS

NT_HF_DATASET  = "InstaDeepAI/nucleotide_transformer_downstream_tasks"
GUE_HF_DATASET = "leannmlindsey/GUE"

NT_TASK_ALIASES = {
    "splice_sites_acceptor": "splice_sites_acceptors",
    "splice_sites_donor": "splice_sites_donors",
}

# GUE+ EPI — Enhancer-Promoter Interaction (DNABERT-2 extended benchmark)
# 6 datasets, one per cell line, sequence length 5000 bp.
# Source: MAGICS-LAB/DNABERT_2 GitHub (manual download required).
# Directory layout expected under --gue-plus-dir:
#   <gue_plus_dir>/EPI/0/train.csv  dev.csv  test.csv
#                       1/...
#                       ...
#                       5/...
# Each CSV: "sequence,label" header, binary classification (0/1).
#
# Cell line mapping used by the DNABERT-2 GUE_v2.zip release:
#   GM12878, HeLa-S3, HUVEC, IMR90, K562, NHEK
GUE_PLUS_EPI_BENCHMARKS = [
    ("epi_0", 2, "EPI-GM12878", "gue+"),
    ("epi_1", 2, "EPI-HeLa-S3", "gue+"),
    ("epi_2", 2, "EPI-HUVEC",   "gue+"),
    ("epi_3", 2, "EPI-IMR90",   "gue+"),
    ("epi_4", 2, "EPI-K562",    "gue+"),
    ("epi_5", 2, "EPI-NHEK",    "gue+"),
]

# Maps task_key → subdirectory name inside <gue_plus_dir>/EPI/
# Override with --epi-subdir-names if your download uses different names.
_EPI_SUBDIR_DEFAULT = {
    "epi_0": "GM12878", "epi_1": "HeLa-S3", "epi_2": "HUVEC",
    "epi_3": "IMR90",   "epi_4": "K562",    "epi_5": "NHEK",
}
_epi_subdir_map: dict = {}   # populated from args at runtime


# ── Model spec ────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    path: str
    mode: str   # "causal" | "bidir" | "encoder"

    @staticmethod
    def parse(spec: str) -> "ModelSpec":
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model spec '{spec}'. "
                "Expected: name:path:mode  (e.g. M0:dnagpt/human_gpt2-v1:causal)"
            )
        name, path, mode = parts
        if mode not in ("causal", "bidir", "encoder"):
            raise ValueError(f"mode must be 'causal' | 'bidir' | 'encoder', got '{mode}'")
        return ModelSpec(name=name, path=path, mode=mode)


# ── Data loading ──────────────────────────────────────────────────────────────

_nt_cache:  dict = {}
_gue_cache: dict = {}
_gb_root = os.environ.get("GENOMIC_BENCHMARKS_DIR", os.path.expanduser("~/.genomic_benchmarks"))
_benchmark_cache_dir = "./cache/benchmarks"

def _gb_pickle_path(task_key: str) -> str:
    return os.path.join(_benchmark_cache_dir, f"gb_{task_key}.pkl")

def _gb_task_present(task_key: str) -> bool:
    task_dir = os.path.join(_gb_root, task_key)
    return os.path.isdir(task_dir)

def _load_gb(task_key: str) -> tuple[list[str], list, list[str], list]:
    """Load a Genomics Benchmarks task via the genomic-benchmarks package."""
    cache_path = _gb_pickle_path(task_key)
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.dataset_getters.pytorch_datasets import GenomicClfDataset

    if not _gb_task_present(task_key):
        download_dataset(task_key, version=0)

    def _unpack(split):
        ds = GenomicClfDataset(task_key, split=split)
        seqs, labels = zip(*[(seq, label) for seq, label in ds])
        return list(seqs), list(labels)

    train_seqs, train_labels = _unpack("train")
    test_seqs,  test_labels  = _unpack("test")
    payload = (train_seqs, train_labels, test_seqs, test_labels)
    os.makedirs(_benchmark_cache_dir, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def _load_nt(task_key: str) -> tuple[list[str], list, list[str], list]:
    """Load one NT downstream task from HuggingFace Hub (filtered by task column)."""
    if "ds" not in _nt_cache:
        from datasets import load_dataset
        _nt_cache["ds"] = load_dataset(NT_HF_DATASET)
    ds = _nt_cache["ds"]
    hf_task_key = NT_TASK_ALIASES.get(task_key, task_key)

    def _filter(split):
        rows = ds[split].filter(lambda x: x["task"] == hf_task_key)
        return [ex["sequence"] for ex in rows], [ex["label"] for ex in rows]

    train_seqs, train_labels = _filter("train")
    test_seqs,  test_labels  = _filter("test")
    return train_seqs, train_labels, test_seqs, test_labels


def _validate_benchmark_splits(
    task_key: str,
    source: str,
    train_seqs: list[str],
    train_labels: list,
    eval_seqs: list[str],
    eval_labels: list,
) -> None:
    split_label = "dev" if source in ("gue", "gue+") else "test"
    if len(train_seqs) != len(train_labels):
        raise ValueError(
            f"{task_key}: train sequence/label count mismatch "
            f"({len(train_seqs)} vs {len(train_labels)})"
        )
    if len(eval_seqs) != len(eval_labels):
        raise ValueError(
            f"{task_key}: {split_label} sequence/label count mismatch "
            f"({len(eval_seqs)} vs {len(eval_labels)})"
        )
    if not train_seqs or not eval_seqs:
        hint = ""
        if source == "nt":
            hf_task_key = NT_TASK_ALIASES.get(task_key, task_key)
            hint = f" Queried HF task='{hf_task_key}'. Check NT_TASK_ALIASES if the dataset schema changed."
        raise ValueError(
            f"{task_key}: loaded empty benchmark split "
            f"(train={len(train_seqs)}, {split_label}={len(eval_seqs)}).{hint}"
        )


def _crop_sequence(seq: str, max_bp: int, mode: str = "center", anchor: int | None = None) -> str:
    """Crop a DNA sequence to at most max_bp base pairs."""
    if len(seq) <= max_bp:
        return seq
    if mode == "junction":
        if anchor is None:
            anchor = len(seq) // 2
        start = anchor - (max_bp // 2)
        start = max(0, min(start, len(seq) - max_bp))
    elif mode == "center":
        start = (len(seq) - max_bp) // 2
    else:
        raise ValueError(f"Unknown crop mode '{mode}'. Expected: center | junction")
    return seq[start : start + max_bp]


def _is_acgt(seq: str) -> bool:
    return all(base in "ACGT" for base in seq)


def _load_gue_plus(
    task_key: str,
    gue_plus_dir: str,
    crop_bp: int = 4096,
    crop_mode: str = "center",
    filter_non_acgt: bool = False,
    reverse_order: bool = False,
) -> tuple[list[str], list, list[str], list]:
    """Load one GUE+ EPI task from locally downloaded DNABERT-2 files.

    Expected directory layout:
        <gue_plus_dir>/EPI/<subdir>/train.csv
                                   dev.csv
                                   test.csv
    CSV format:
      - center crop: "sequence,label" or "seq,label"
      - junction crop: requires separate "enhancer" and "promoter" columns

    Args:
        task_key    : one of "epi_0" … "epi_5"
        gue_plus_dir: root directory containing the EPI/ folder
        crop_bp     : centre-crop long sequences to this many bp before
                      tokenisation (default 4096 = 1024 tokens × 4 bp/token)
    """
    import csv as _csv

    subdir = _epi_subdir_map.get(task_key, _EPI_SUBDIR_DEFAULT.get(task_key, task_key))
    task_dir = os.path.join(gue_plus_dir, "EPI", subdir)

    if not os.path.isdir(task_dir):
        raise FileNotFoundError(
            f"GUE+ EPI directory not found: {task_dir}\n"
            f"  Download from https://github.com/MAGICS-LAB/DNABERT_2\n"
            f"  Then set --gue-plus-dir to the parent of the EPI/ folder.\n"
            f"  If subdirs use cell-line names instead of 0–5, pass\n"
            f"  --epi-subdir-names H1 HCT116 GM12878 K562 WTC11 IMR90"
        )

    def _read_split(split_name: str):
        path = os.path.join(task_dir, f"{split_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        seqs, labels = [], []
        dropped = 0
        dropped_empty_parts = 0
        with open(path, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                seq = row.get("sequence") or row.get("seq") or ""
                anchor = None
                has_parts = "enhancer" in row and "promoter" in row

                if crop_mode == "junction" and not has_parts:
                    raise ValueError(
                        f"EPI junction crop requires 'enhancer' and 'promoter' columns, "
                        f"but {path} does not provide them. Current row keys: {sorted(row.keys())}. "
                        f"Use --epi-crop-mode center for pre-concatenated sequence-only CSVs."
                    )

                if reverse_order and not has_parts:
                    raise ValueError(
                        f"--epi-reverse-order requires separate 'enhancer' and 'promoter' columns, "
                        f"but {path} does not provide them. Current row keys: {sorted(row.keys())}."
                    )

                if has_parts:
                    enhancer = row["enhancer"].strip().upper()
                    promoter = row["promoter"].strip().upper()

                    if (crop_mode == "junction" or reverse_order) and (not enhancer or not promoter):
                        dropped_empty_parts += 1
                        continue

                if not seq and has_parts:
                    if reverse_order:
                        seq = promoter + enhancer
                        anchor = len(promoter)
                    else:
                        seq = enhancer + promoter
                        anchor = len(enhancer)
                seq = _crop_sequence(seq.strip().upper(), crop_bp, crop_mode, anchor)
                if filter_non_acgt and not _is_acgt(seq):
                    dropped += 1
                    continue
                labels.append(int(row["label"]))
                seqs.append(seq)
        if dropped_empty_parts:
            print(
                f"       [empty-enh/prom] dropped {dropped_empty_parts} {split_name} rows with "
                f"empty enhancer/promoter fields"
            )
        if dropped:
            print(f"       [filter-n] dropped {dropped} {split_name} rows with non-ACGT bases")
        if not seqs:
            raise ValueError(
                f"No usable rows remain in {path} after filtering. "
                f"Check enhancer/promoter fields and filtering settings."
            )
        return seqs, labels

    tr_s, tr_l = _read_split("train")
    te_s, te_l = _read_split("dev")    # use dev (public labels), not test
    return tr_s, tr_l, te_s, te_l


def _load_gue(task_key: str) -> tuple[list[str], list, list[str], list]:
    """Load one GUE task from leannmlindsey/GUE on HuggingFace Hub.

    Train on the 'train' split; evaluate on the 'dev' split (public labels).
    The 'test' split labels are not publicly released, so we follow the
    standard practice of using dev as the held-out evaluation set for
    linear probe experiments.

    Columns: sequence (str), label (int).
    """
    from datasets import load_dataset
    if task_key not in _gue_cache:
        _gue_cache[task_key] = load_dataset(GUE_HF_DATASET, name=task_key)
    ds = _gue_cache[task_key]

    train_seqs   = ds["train"]["sequence"]
    train_labels = ds["train"]["label"]
    dev_seqs     = ds["dev"]["sequence"]
    dev_labels   = ds["dev"]["label"]
    return list(train_seqs), list(train_labels), list(dev_seqs), list(dev_labels)


def load_benchmark_data(
    task_key: str,
    source: str,
    gue_plus_dir: str = "",
    crop_bp: int = 4096,
    crop_mode: str = "center",
    filter_non_acgt: bool = False,
    epi_reverse_order: bool = False,
) -> tuple[list[str], list, list[str], list]:
    if source == "gb":
        payload = _load_gb(task_key)
    elif source == "nt":
        payload = _load_nt(task_key)
    elif source == "gue":
        payload = _load_gue(task_key)
    elif source == "gue+":
        payload = _load_gue_plus(
            task_key,
            gue_plus_dir,
            crop_bp=crop_bp,
            crop_mode=crop_mode,
            filter_non_acgt=filter_non_acgt,
            reverse_order=epi_reverse_order,
        )
    else:
        raise ValueError(f"Unknown source '{source}'")
    _validate_benchmark_splits(task_key, source, *payload)
    return payload


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    pooling: str = "mean",
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Pool token embeddings into one sequence embedding."""
    mask = attention_mask.to(dtype=hidden.dtype)

    if pooling == "mean":
        mask_3d = mask.unsqueeze(-1)
        return (hidden * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1e-9)

    if pooling == "weighted_mean":
        positions = torch.arange(1, hidden.size(1) + 1, device=hidden.device, dtype=hidden.dtype)
        weights = positions.unsqueeze(0) * mask
        weights_3d = weights.unsqueeze(-1)
        return (hidden * weights_3d).sum(dim=1) / weights_3d.sum(dim=1).clamp(min=1e-9)

    if pooling == "last":
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]

    if pooling == "cls":
        first_idx = attention_mask.long().argmax(dim=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, first_idx]

    if pooling == "eos":
        if input_ids is not None and eos_token_id is not None:
            eos_mask = (input_ids == eos_token_id) & attention_mask.bool()
            has_eos = eos_mask.any(dim=1)
            # Pick the last EOS if multiple are present.
            eos_positions = eos_mask.long() * torch.arange(
                1, input_ids.size(1) + 1, device=input_ids.device
            ).unsqueeze(0)
            eos_idx = eos_positions.argmax(dim=1).clamp(min=1) - 1
            last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
            idx = torch.where(has_eos, eos_idx, last_idx)
        else:
            idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, idx]

    raise ValueError(f"Unknown pooling '{pooling}'")


def encode_sequences(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    pooling: str = "mean",
    normalize: bool = True,
    desc: str = "Encoding",
) -> np.ndarray:
    """Pool + L2-normalise embeddings. Returns float32 array (N, D)."""
    model.eval()
    all_embeddings = []

    if not sequences:
        raise ValueError(f"{desc}: no sequences to encode")

    for i in tqdm(range(0, len(sequences), batch_size), desc=desc, leave=False):
        batch = sequences[i : i + batch_size]
        with torch.inference_mode():
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding="longest",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # Decoder (GPT-2): use model.transformer submodule
            # Encoder (BERT/ESM): call model directly
            if hasattr(model, "transformer"):
                out = model.transformer(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
            else:
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif isinstance(out, tuple):
                hidden = out[0]
            else:
                raise TypeError(
                    f"Unsupported model output type {type(out)!r}; "
                    "expected an object with last_hidden_state or a tuple whose "
                    "first element is hidden states."
                )
            pooled = pool_hidden_states(
                hidden=hidden,
                attention_mask=enc["attention_mask"],
                input_ids=enc["input_ids"],
                pooling=pooling,
                eos_token_id=tokenizer.eos_token_id,
            )
            if normalize:
                pooled = F.normalize(pooled, dim=-1)

        all_embeddings.append(pooled.cpu().float().numpy())
        del enc, out, hidden, pooled

    return np.concatenate(all_embeddings, axis=0)


# ── Linear probe ─────────────────────────────────────────────────────────────

def run_linear_probe(
    train_emb: np.ndarray, train_labels: list,
    test_emb:  np.ndarray, test_labels:  list,
    seed: int = 42,
    debug_name: str | None = None,
    standardize: bool = False,
    class_weight: str | None = None,
) -> dict:
    """Train a logistic regression probe and return accuracy, F1 (macro), and MCC.

    All three metrics are returned for every task so that:
      - GB / NT tables display accuracy (conventional for those benchmarks).
      - GUE tables display F1 and MCC (matching DNABERT-2 paper convention).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.preprocessing import StandardScaler

    le      = LabelEncoder()
    train_y = le.fit_transform(train_labels)
    test_y  = le.transform(test_labels)

    if standardize:
        scaler = StandardScaler()
        train_emb = scaler.fit_transform(train_emb)
        test_emb = scaler.transform(test_emb)

    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        C=1.0,
        class_weight=class_weight,
        random_state=seed,
    )
    clf.fit(train_emb, train_y)
    preds = clf.predict(test_emb)
    pred_counts = np.bincount(preds, minlength=len(le.classes_)).astype(int)
    label_counts = np.bincount(test_y, minlength=len(le.classes_)).astype(int)
    if debug_name:
        train_norm = np.linalg.norm(train_emb, axis=1)
        test_norm = np.linalg.norm(test_emb, axis=1)
        print(
            f"       [probe] {debug_name}: "
            f"test_labels={label_counts.tolist()} preds={pred_counts.tolist()} "
            f"train_norm={train_norm.mean():.3g}+/-{train_norm.std():.3g} "
            f"test_norm={test_norm.mean():.3g}+/-{test_norm.std():.3g}"
        )

    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "f1":       float(f1_score(test_y, preds, average="macro")),
        "mcc":      float(matthews_corrcoef(test_y, preds)),
        "test_label_counts": label_counts.tolist(),
        "pred_label_counts": pred_counts.tolist(),
        "probe_standardize": bool(standardize),
        "probe_class_weight": class_weight or "none",
    }


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(spec: ModelSpec, device: str, dtype):
    # Resolve relative paths to absolute so huggingface_hub doesn't reject './'
    path = os.path.abspath(spec.path) if os.path.exists(spec.path) else spec.path
    print(f"  Loading {spec.name}  ({path}, mode={spec.mode})")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if spec.mode == "encoder":
        # Bidirectional encoder (e.g. NT-500M / ESM): load as plain AutoModel
        from transformers import AutoConfig
        enc_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if not hasattr(enc_config, "pad_token_id") or enc_config.pad_token_id is None:
            enc_config.pad_token_id = tokenizer.pad_token_id or 0
        # Some remote-code encoder models (notably DNABERT-2) construct tensors
        # inside __init__. Newer Transformers/PyTorch combinations can otherwise
        # instantiate parts of the model on the meta device during loading and
        # crash before weights are materialized. Force a conservative CPU load.
        with torch.device("cpu"):
            model = AutoModel.from_pretrained(
                path,
                config=enc_config,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                _fast_init=False,
            )
        model = model.to(device=device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="eager",
        )
        if spec.mode == "bidir":
            model = patch_to_bidirectional(model)

    if spec.mode != "encoder":
        model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Parameters : {n_params:.1f}M  |  vocab: {len(tokenizer):,}")
    return model, tokenizer


# ── Results tables ────────────────────────────────────────────────────────────

def _metric_val(task_metrics: dict, metric: str) -> float:
    """Extract a scalar from a per-task metrics dict (or nan if missing)."""
    if isinstance(task_metrics, dict):
        return task_metrics.get(metric, float("nan"))
    # Backward-compat: plain float stored (shouldn't happen in new code)
    return float(task_metrics)


def print_results_table(
    results: dict,
    benchmarks: list[tuple],
    title: str,
    metric: str = "accuracy",
):
    """Print a results table using the specified primary metric (accuracy / f1 / mcc)."""
    short_names  = [b[2] for b in benchmarks]
    model_names  = list(results.keys())
    col_w        = max(max(len(n) for n in model_names), 6)
    task_w       = 13

    print(f"\n{title}  [metric: {metric}]")
    header = (
        f"{'Model':<{col_w}}  "
        + "  ".join(f"{n:>{task_w}}" for n in short_names)
        + f"  {'Avg':>{task_w}}"
    )
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)

    for model_name, task_metrics_map in results.items():
        vals = [_metric_val(task_metrics_map.get(b[0], {}), metric) for b in benchmarks]
        avg  = np.nanmean(vals)
        row  = f"{model_name:<{col_w}}  "
        row += "  ".join(f"{v * 100:>{task_w}.2f}" for v in vals)
        row += f"  {avg * 100:>{task_w}.2f}"
        print(row)

    print(sep)
    unit = "%" if metric in ("accuracy", "f1") else "(×100%)"
    print(f"({metric} {unit}, linear probe, train→dev for GUE / train→test for GB/NT)")


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DNA-LLM2Vec Step 4: Evaluation")

    p.add_argument("--models", nargs="+", required=True, metavar="name:path:mode",
                   help="Model specs: name:path:mode. mode = causal | bidir | encoder.")

    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--gb-only",       action="store_true",
                       help="Run Genomics Benchmarks only (8 GB tasks).")
    suite.add_argument("--nt-only",       action="store_true",
                       help="Run NT representative subset only (4 NT tasks).")
    suite.add_argument("--gue-only",      action="store_true",
                       help="Run GUE benchmark only (18 tasks, F1+MCC metrics).")
    suite.add_argument("--gue-plus-only", action="store_true",
                       help="Run GUE+ EPI benchmark only (6 tasks, 5000 bp sequences).")

    p.add_argument("--output",     default="./eval_results/results.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--gb-root", default=os.environ.get("GENOMIC_BENCHMARKS_DIR", os.path.expanduser("~/.genomic_benchmarks")),
                   help="Root directory for genomic_benchmarks task folders. If a task is missing here, it will be downloaded.")
    p.add_argument("--benchmark-cache-dir", default="./cache/benchmarks",
                   help="Directory for serialized benchmark caches (used to avoid repeatedly scanning GB small files).")
    p.add_argument(
        "--pooling",
        choices=("mean", "weighted_mean", "last", "cls", "eos"),
        default="mean",
        help="Embedding pooling strategy. mean keeps previous Step4 behaviour; "
             "last and weighted_mean match common LLM2Vec-style evaluation choices.",
    )
    p.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization of pooled sequence embeddings before the linear probe.",
    )
    p.add_argument(
        "--probe-standardize",
        action="store_true",
        help="Standardize embedding dimensions with train-split statistics before the linear probe.",
    )
    p.add_argument(
        "--probe-class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Class weighting for the logistic-regression probe.",
    )
    p.add_argument("--run-name", default=None,
                   help="Optional W&B run name. If omitted, a descriptive name is generated.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed recorded in W&B/config and used by the linear probe.")
    p.add_argument("--repeat-index", type=int, default=None,
                   help="Optional repeat id for repeated experiments, e.g. 1, 2, 3.")
    p.add_argument("--no-wandb",   action="store_true")

    # GUE+ EPI options
    p.add_argument(
        "--gue-plus-dir", default="",
        help="Root directory of the locally downloaded GUE+ data "
             "(parent of the EPI/ folder). If provided during the default "
             "full evaluation, the 6 EPI tasks are appended to GB+NT+GUE.",
    )
    p.add_argument(
        "--epi-subdir-names", nargs=6, default=None,
        metavar=("S0", "S1", "S2", "S3", "S4", "S5"),
        help="Override EPI subdirectory names for epi_0…epi_5. "
             "Default: GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK.",
    )
    p.add_argument(
        "--epi-crop-bp", type=int, default=4096,
        help="Centre-crop EPI sequences to this many bp before tokenisation "
             "(default: 4096 = 1024 tokens × 4 bp/token). "
             "5000 bp sequences lose ~452 bp from each end.",
    )
    p.add_argument(
        "--epi-crop-mode", choices=("center", "junction"), default="center",
        help="How to crop GUE+ EPI sequences: center uses the sequence midpoint; "
             "junction centers on the enhancer/promoter boundary and requires "
             "separate enhancer/promoter columns in the CSV.",
    )
    p.add_argument(
        "--epi-reverse-order", action="store_true",
        help="For GUE+ EPI rows with separate enhancer/promoter columns, "
             "concatenate promoter+enhancer instead of enhancer+promoter. "
             "This flag is invalid for sequence-only CSVs.",
    )
    p.add_argument(
        "--filter-n", action="store_true",
        help="For GUE+ EPI, drop rows whose cropped sequence contains any non-ACGT base.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    specs = [ModelSpec.parse(s) for s in args.models]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    global _gb_root, _benchmark_cache_dir
    _gb_root = os.path.expanduser(args.gb_root)
    _benchmark_cache_dir = args.benchmark_cache_dir

    # Populate EPI subdir map from args (or keep defaults)
    global _epi_subdir_map
    if args.epi_subdir_names:
        keys = [f"epi_{i}" for i in range(6)]
        _epi_subdir_map = dict(zip(keys, args.epi_subdir_names))
    else:
        _epi_subdir_map = dict(_EPI_SUBDIR_DEFAULT)

    if args.gb_only:
        active = GB_BENCHMARKS
    elif args.nt_only:
        active = NT_BENCHMARKS
    elif args.gue_only:
        active = GUE_BENCHMARKS
    elif args.gue_plus_only:
        if not args.gue_plus_dir:
            print("ERROR: --gue-plus-only requires --gue-plus-dir <path>")
            sys.exit(1)
        active = GUE_PLUS_EPI_BENCHMARKS
    else:
        active = ALL_BENCHMARKS + (GUE_PLUS_EPI_BENCHMARKS if args.gue_plus_dir else [])

    if any(b[3] == "gue+" for b in active) and args.max_length < args.epi_crop_bp:
        print(
            f"  [warn] --max-length {args.max_length} is smaller than "
            f"--epi-crop-bp {args.epi_crop_bp}; tokenizers with near 1 bp/token "
            "will truncate EPI crops before pooling."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 68)
    print("DNA-LLM2Vec  |  Step 4 — Linear Probe Evaluation")
    print("=" * 68)
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    gb_n  = sum(1 for b in active if b[3] == "gb")
    nt_n  = sum(1 for b in active if b[3] == "nt")
    gue_n  = sum(1 for b in active if b[3] == "gue")
    guep_n = sum(1 for b in active if b[3] == "gue+")
    print(f"  Tasks      : {gb_n} GB  +  {nt_n} NT  +  {gue_n} GUE  +  {guep_n} GUE+ EPI  =  {len(active)} total")
    print(f"  Models     : {[s.name for s in specs]}")
    print(f"  Max length : {args.max_length}")
    print(f"  Pooling    : {args.pooling}")
    print(f"  L2 norm    : {not args.no_l2_normalize}")
    print(f"  Probe      : standardize={args.probe_standardize}  class_weight={args.probe_class_weight}")
    print(f"  Seed       : {args.seed}")
    if args.repeat_index is not None:
        print(f"  Repeat     : {args.repeat_index}")
    print("=" * 68)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "dna_foundation"
        os.environ["WANDB_ENTITY"]  = "liangyuan-edin-queen-mary-university-of-london"
        try:
            import wandb
            model_label = "_".join(s.name for s in specs)
            suite_label = (
                "gb" if args.gb_only else
                "nt" if args.nt_only else
                "gue" if args.gue_only else
                "gueplus" if args.gue_plus_only else
                "all"
            )
            repeat_suffix = f"_r{args.repeat_index}" if args.repeat_index is not None else ""
            run_name = args.run_name or f"step4_{suite_label}_{model_label}_{args.pooling}_s{args.seed}{repeat_suffix}"
            wandb.init(
                job_type="eval",
                name=run_name,
                config={
                    "models":     args.models,
                    "max_length": args.max_length,
                    "pooling":    args.pooling,
                    "l2_normalize": not args.no_l2_normalize,
                    "probe_standardize": args.probe_standardize,
                    "probe_class_weight": args.probe_class_weight,
                    "seed":       args.seed,
                    "repeat_index": args.repeat_index,
                    "gb_tasks":   gb_n,
                    "nt_tasks":   nt_n,
                    "gue_tasks":  gue_n,
                    "gue_plus_epi_tasks": guep_n,
                },
            )
        except Exception as e:
            print(f"  [warn] W&B init failed: {e}")

    # ── Load datasets once ────────────────────────────────────────────────────
    print("\n[1/3] Loading datasets")
    benchmark_data = {}
    for task_key, _, display_name, source in active:
        print(f"  [{source.upper()}] {display_name} ({task_key})")
        train_seqs, train_labels, test_seqs, test_labels = load_benchmark_data(
            task_key, source,
            gue_plus_dir=args.gue_plus_dir,
            crop_bp=args.epi_crop_bp,
            crop_mode=args.epi_crop_mode,
            filter_non_acgt=args.filter_n,
            epi_reverse_order=args.epi_reverse_order,
        )
        benchmark_data[task_key] = (train_seqs, train_labels, test_seqs, test_labels)
        split_label = "dev" if source in ("gue", "gue+") else "test"
        extra = f"  [{args.epi_crop_mode}-cropped to {args.epi_crop_bp} bp" if source == "gue+" else ""
        if source == "gue+":
            extra += ", promoter+enhancer" if args.epi_reverse_order else ""
            extra += ", filter-n" if args.filter_n else ""
            extra += "]"
        print(f"       train={len(train_seqs):,}  {split_label}={len(test_seqs):,}{extra}")

    # ── Evaluate each model ───────────────────────────────────────────────────
    print("\n[2/3] Evaluating models")
    results = {}   # {model_name: {task_key: {"accuracy": float, "f1": float, "mcc": float}}}

    for spec in specs:
        print(f"\n── {spec.name} ──────────────────────────────────────────────")
        model, tokenizer = load_model(spec, device, dtype)
        results[spec.name] = {}

        for task_key, _, display_name, source in active:
            train_seqs, train_labels, test_seqs, test_labels = benchmark_data[task_key]

            train_emb = encode_sequences(model, tokenizer, train_seqs,
                                         args.batch_size, args.max_length, device,
                                         pooling=args.pooling,
                                         normalize=not args.no_l2_normalize,
                                         desc=f"  {display_name} train")
            test_emb  = encode_sequences(model, tokenizer, test_seqs,
                                         args.batch_size, args.max_length, device,
                                         pooling=args.pooling,
                                         normalize=not args.no_l2_normalize,
                                         desc=f"  {display_name} test ")

            metrics = run_linear_probe(
                train_emb,
                train_labels,
                test_emb,
                test_labels,
                seed=args.seed,
                debug_name=display_name if source == "gue+" else None,
                standardize=args.probe_standardize,
                class_weight=None if args.probe_class_weight == "none" else args.probe_class_weight,
            )
            results[spec.name][task_key] = metrics

            # Display all task metrics for every suite so later comparisons do not
            # depend on a single hard-coded benchmark convention.
            if source in ("gue", "gue+"):
                print(f"  {display_name:<28}  F1={metrics['f1']*100:.2f}%  MCC={metrics['mcc']*100:.2f}%")
            else:
                print(
                    f"  {display_name:<28}  acc={metrics['accuracy']*100:.2f}%  "
                    f"F1={metrics['f1']*100:.2f}%  MCC={metrics['mcc']*100:.2f}%"
                )

        import gc
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate averages ────────────────────────────────────────────────────
    print("\n[3/3] Results")

    gb_active   = [b for b in active if b[3] == "gb"]
    nt_active   = [b for b in active if b[3] == "nt"]
    gue_active  = [b for b in active if b[3] == "gue"]
    guep_active = [b for b in active if b[3] == "gue+"]

    for model_name, task_metrics_map in results.items():
        if gb_active:
            results[model_name]["avg_gb_accuracy"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "accuracy") for b in gb_active]
            ))
            results[model_name]["avg_gb_f1"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "f1") for b in gb_active]
            ))
            results[model_name]["avg_gb_mcc"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "mcc") for b in gb_active]
            ))
        if nt_active:
            results[model_name]["avg_nt_accuracy"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "accuracy") for b in nt_active]
            ))
            results[model_name]["avg_nt_f1"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "f1") for b in nt_active]
            ))
            results[model_name]["avg_nt_mcc"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "mcc") for b in nt_active]
            ))
        if gue_active:
            results[model_name]["avg_gue_f1"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "f1") for b in gue_active]
            ))
            results[model_name]["avg_gue_mcc"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "mcc") for b in gue_active]
            ))
        if guep_active:
            results[model_name]["avg_epi_f1"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "f1") for b in guep_active]
            ))
            results[model_name]["avg_epi_mcc"] = float(np.nanmean(
                [_metric_val(task_metrics_map.get(b[0], {}), "mcc") for b in guep_active]
            ))

    if gb_active:
        print_results_table(
            results, gb_active,
            "Genomics Benchmarks (Grevsova et al., 2023)",
            metric="accuracy",
        )
        print_results_table(
            results, gb_active,
            "Genomics Benchmarks (Grevsova et al., 2023)",
            metric="f1",
        )
        print_results_table(
            results, gb_active,
            "Genomics Benchmarks (Grevsova et al., 2023)",
            metric="mcc",
        )
    if nt_active:
        print_results_table(
            results, nt_active,
            "Nucleotide Transformer Downstream Tasks",
            metric="accuracy",
        )
        print_results_table(
            results, nt_active,
            "Nucleotide Transformer Downstream Tasks",
            metric="f1",
        )
        print_results_table(
            results, nt_active,
            "Nucleotide Transformer Downstream Tasks",
            metric="mcc",
        )
    if gue_active:
        print_results_table(
            results, gue_active,
            "GUE Benchmark (DNABERT-2, 18 tasks)",
            metric="f1",
        )
        print_results_table(
            results, gue_active,
            "GUE Benchmark (DNABERT-2, 18 tasks)",
            metric="mcc",
        )
    if guep_active:
        print_results_table(
            results, guep_active,
            "GUE+ EPI Benchmark (Enhancer-Promoter Interaction, 6 tasks)",
            metric="f1",
        )
        print_results_table(
            results, guep_active,
            "GUE+ EPI Benchmark (Enhancer-Promoter Interaction, 6 tasks)",
            metric="mcc",
        )

    # ── Print averages ────────────────────────────────────────────────────────
    print("\nSummary averages:")
    for model_name in results:
        m = results[model_name]
        parts = []
        if "avg_gb_accuracy" in m:
            parts.append(f"GB acc={m['avg_gb_accuracy']*100:.2f}%")
        if "avg_gb_f1" in m:
            parts.append(f"GB F1={m['avg_gb_f1']*100:.2f}%")
        if "avg_gb_mcc" in m:
            parts.append(f"GB MCC={m['avg_gb_mcc']*100:.2f}%")
        if "avg_nt_accuracy" in m:
            parts.append(f"NT acc={m['avg_nt_accuracy']*100:.2f}%")
        if "avg_nt_f1" in m:
            parts.append(f"NT F1={m['avg_nt_f1']*100:.2f}%")
        if "avg_nt_mcc" in m:
            parts.append(f"NT MCC={m['avg_nt_mcc']*100:.2f}%")
        if "avg_gue_f1" in m:
            parts.append(f"GUE F1={m['avg_gue_f1']*100:.2f}%")
        if "avg_gue_mcc" in m:
            parts.append(f"GUE MCC={m['avg_gue_mcc']*100:.2f}%")
        if "avg_epi_f1" in m:
            parts.append(f"EPI F1={m['avg_epi_f1']*100:.2f}%")
        if "avg_epi_mcc" in m:
            parts.append(f"EPI MCC={m['avg_epi_mcc']*100:.2f}%")
        print(f"  {model_name}: " + "  |  ".join(parts))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # ── W&B summary ──────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            if wandb.run is not None:
                for model_name, m in results.items():
                    for k, v in m.items():
                        if k.startswith("avg_") and isinstance(v, float):
                            wandb.summary[f"{model_name}/{k}"] = v
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
