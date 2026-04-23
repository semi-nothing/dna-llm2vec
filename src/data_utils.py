"""
DNA-LLM2Vec  |  Data Utilities
==============================
Handles loading genomic sequences from:
  1. A local FASTA file  (e.g. hg38.fa)
  2. A HuggingFace dataset  (e.g. 'dnagpt/human_genome')

All sequences are chunked into fixed-length windows and tokenised.
"""

import re
from typing import Generator

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


# ── FASTA parsing ─────────────────────────────────────────────────────────────

VALID_BASES = re.compile(r"[^ACGT]")   # matches any non-ACGT character (N, ambiguous, etc.)
HAS_NON_ACGT = VALID_BASES  # alias used for filter_n checks (same pattern)


def _iter_fasta(fasta_path: str, strip_n: bool = True) -> Generator[str, None, None]:
    """
    Yield one full chromosome/contig sequence at a time from a FASTA file.
    Strips headers, converts to uppercase.

    Args:
        fasta_path : path to .fa or .fa.gz file
        strip_n    : if True (default), remove all non-ACGT characters in-place,
                     joining across N-regions (legacy behaviour).
                     if False, yield the raw uppercase sequence including N/ambiguous
                     bases — callers that set filter_n=True will then drop chunks
                     that contain any non-ACGT character, which correctly avoids
                     artificial sequence junctions across N-gaps (DNABERT-2 style).
    """
    import gzip

    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    seq_chunks = []

    def _emit(chunks):
        seq = "".join(chunks).upper()
        if strip_n:
            seq = VALID_BASES.sub("", seq)
        return seq if seq else None

    with opener(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chunks:
                    seq = _emit(seq_chunks)
                    if seq:
                        yield seq
                seq_chunks = []
            else:
                seq_chunks.append(line)

    # Last entry
    if seq_chunks:
        seq = _emit(seq_chunks)
        if seq:
            yield seq


def _chunk_sequence(seq: str, chunk_size: int, stride: int) -> list[str]:
    """
    Split a long sequence into overlapping windows of `chunk_size` with
    step `stride`. Shorter windows at the tail are discarded.

    Args:
        seq        : raw DNA string (ACGT only)
        chunk_size : token budget in base pairs (before BPE compression)
        stride     : step between windows; stride < chunk_size gives overlap
                     which increases effective data size and smooths boundaries
    """
    chunks = []
    for start in range(0, len(seq) - chunk_size + 1, stride):
        chunks.append(seq[start : start + chunk_size])
    return chunks


def load_fasta_dataset(
    fasta_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    stride: int = 0,
    val_fraction: float = 0.01,
    seed: int = 42,
    filter_n: bool = False,
) -> DatasetDict:
    """
    Load a FASTA file and return a train/validation DatasetDict.

    Args:
        fasta_path   : path to .fa or .fa.gz file
        tokenizer    : DNAGPT tokenizer (BPE)
        max_length   : model context length in tokens; sequences are chunked
                       to ~max_length * avg_chars_per_token base pairs
        stride       : sliding window stride in base pairs
        val_fraction : fraction of chunks reserved for validation
        seed         : random seed for the train/val split
        filter_n     : if True, drop any chunk that contains a non-ACGT character
                       (N or other ambiguous bases) — matches DNABERT-2 preprocessing.
                       When False (default), non-ACGT characters are stripped in-place
                       before chunking (legacy behaviour, creates artificial junctions
                       across chromosomal N-gaps).

    Notes:
        BPE tokenisation compresses sequence: a 512-token window corresponds
        to roughly 1–3 kb of DNA depending on k-mer frequency. Use
        chunk_size = max_length * 4 as a conservative estimate.
    """
    print(f"Loading FASTA: {fasta_path}")
    if filter_n:
        print("  filter_n=True  →  chunks containing N/ambiguous bases will be discarded")

    # BPE compresses DNA: estimate 4 bp per token (conservative for DNAGPT BPE)
    bp_per_token  = 4
    chunk_size_bp = max_length * bp_per_token

    # stride=0 → non-overlapping windows (safe default for large genomes)
    effective_stride = stride if stride > 0 else chunk_size_bp

    # ── Generator-based loading ───────────────────────────────────────────────
    # DO NOT collect all_chunks into a list first.
    # hg38 with chunk_size=2048bp and stride=256bp → ~12M chunks → ~24GB RAM.
    # Dataset.from_generator() streams lazily and writes to an Arrow cache on disk.
    contig_count   = [0]
    chunk_count    = [0]
    filtered_count = [0]

    def sequence_generator():
        # strip_n=False when filter_n: keep N in contig so chunk boundaries
        # that span an N-gap are detected and dropped rather than silently merged.
        for contig_seq in _iter_fasta(fasta_path, strip_n=not filter_n):
            chunks = _chunk_sequence(contig_seq, chunk_size_bp, effective_stride)
            contig_count[0] += 1
            kept = []
            for chunk in chunks:
                if filter_n and VALID_BASES.search(chunk):
                    filtered_count[0] += 1
                else:
                    kept.append(chunk)
            chunk_count[0] += len(kept)
            print(
                f"  Contig {contig_count[0]:>3}  {len(contig_seq):>12,} bp  "
                f"→  {len(kept):>7,} chunks"
                + (f"  (dropped {len(chunks) - len(kept)} N-containing)" if filter_n else "")
            )
            for chunk in kept:
                yield {"sequence": chunk}

    raw_dataset = Dataset.from_generator(sequence_generator)
    print(f"  Total chunks : {chunk_count[0]:,}")
    if filter_n:
        print(f"  Filtered out : {filtered_count[0]:,} chunks with N/ambiguous bases")

    # Tokenise
    def tokenise(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenised = raw_dataset.map(
        tokenise,
        batched=True,
        batch_size=1000,
        remove_columns=["sequence"],
        desc="Tokenising",
    )

    # Train / val split
    splits = tokenised.train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


# ── HuggingFace Hub dataset ───────────────────────────────────────────────────

def load_hub_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    sequence_column: str = "sequence",
    val_fraction: float = 0.01,
    seed: int = 42,
    streaming: bool = False,
) -> DatasetDict:
    """
    Load a sequence dataset from HuggingFace Hub and tokenise it.

    Recommended datasets:
      - 'dnagpt/human_genome'           (large, use streaming=True)
      - 'InstaDeepAI/nucleotide_transformer_downstream_tasks'

    Args:
        dataset_name     : HuggingFace dataset identifier
        tokenizer        : DNAGPT tokenizer
        max_length       : max token length per sample
        sequence_column  : name of the column containing raw DNA strings
        val_fraction     : fraction for validation split
        streaming        : use streaming mode (avoids downloading full dataset)
    """
    print(f"Loading HuggingFace dataset: {dataset_name}")
    raw = load_dataset(dataset_name, streaming=streaming)

    def tokenise(batch):
        return tokenizer(
            batch[sequence_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    if streaming:
        # Streaming: cannot do train_test_split; caller should use a separate val set
        tokenised_train = raw["train"].map(tokenise, batched=True)
        return DatasetDict({"train": tokenised_train})

    tokenised = raw.map(
        tokenise,
        batched=True,
        remove_columns=[sequence_column],
        desc="Tokenising",
    )

    if "validation" not in tokenised:
        splits = tokenised["train"].train_test_split(test_size=val_fraction, seed=seed)
        return DatasetDict({"train": splits["train"], "validation": splits["test"]})

    return tokenised


# ── Mask token setup ──────────────────────────────────────────────────────────

def ensure_mask_token(tokenizer: PreTrainedTokenizer, model) -> bool:
    """
    GPT-2 based tokenizers do not include a [MASK] token.
    This function adds one if absent and resizes the model's token embeddings.

    Returns True if the token was added (model embeddings were resized),
    False if [MASK] was already present.
    """
    if tokenizer.mask_token is not None:
        return False  # already exists

    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model.resize_token_embeddings(len(tokenizer))

    print(f"  Added [MASK] token — vocab size now: {len(tokenizer):,}")
    return True
