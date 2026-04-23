"""
DNA-LLM2Vec | Download GUE+ EPI Data
====================================

Prepare or inspect the GUE+ Enhancer-Promoter Interaction (EPI) benchmark
used by DNABERT-2. The evaluation scripts expect:

  <out-dir>/EPI/<subdir>/train.csv
  <out-dir>/EPI/<subdir>/dev.csv
  <out-dir>/EPI/<subdir>/test.csv

Usage:
  python src/download_gue_plus.py --out-dir ./data/GUE_plus
  python src/download_gue_plus.py --out-dir ./data/GUE_plus --inspect-only
"""

import argparse
import os


GUE_PLUS_GDRIVE_README = "https://github.com/MAGICS-LAB/DNABERT_2/blob/main/README.md"
GUE_PLUS_ZENODO = "https://zenodo.org/records/8399978"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="./data/GUE_plus",
        help="Where to store GUE+ data (default: ./data/GUE_plus).",
    )
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Skip download instructions; just inspect an existing directory layout.",
    )
    return p.parse_args()


def inspect_epi_layout(out_dir: str):
    """Print the EPI layout and suggest flags for step4/step5."""
    epi_dir = os.path.join(out_dir, "EPI")
    if not os.path.isdir(epi_dir):
        print(f"[warn] EPI directory not found: {epi_dir}")
        print("       Download GUE+ first, then re-run with --inspect-only.")
        return

    subdirs = sorted(
        d for d in os.listdir(epi_dir)
        if os.path.isdir(os.path.join(epi_dir, d))
    )

    print(f"\nFound {len(subdirs)} EPI subdirectories under {epi_dir}:")
    for i, name in enumerate(subdirs):
        task_dir = os.path.join(epi_dir, name)
        files = os.listdir(task_dir)
        splits = [f for f in files if f.endswith(".csv")]

        seq_len = "?"
        train_csv = os.path.join(task_dir, "train.csv")
        if os.path.exists(train_csv):
            import csv
            with open(train_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    seq = row.get("sequence") or row.get("seq") or ""
                    if not seq and "enhancer" in row and "promoter" in row:
                        seq = row["enhancer"] + row["promoter"]
                    seq_len = str(len(seq))
                    break

        n_train = "?"
        if os.path.exists(train_csv):
            with open(train_csv) as fh:
                n_train = str(sum(1 for _ in fh) - 1)

        print(
            f"  [{i}] {name:20s}  splits={splits}  "
            f"train_rows={n_train}  seq_len={seq_len} bp"
        )

    print()
    if len(subdirs) == 6:
        names_str = " ".join(subdirs)
        print("Suggested step4 / step5 flags:")
        print(f"  --gue-plus-dir {out_dir}")
        if not all(d.isdigit() for d in subdirs):
            print(f"  --epi-subdir-names {names_str}")
        else:
            print("  (subdir names are 0-5, no --epi-subdir-names needed)")
    else:
        print(f"[warn] Expected 6 EPI subdirs, found {len(subdirs)}.")
        print("       Check your download or set --epi-subdir-names manually.")


def print_download_instructions(out_dir: str):
    """Print manual download instructions for the DNABERT-2 GUE+ data."""
    os.makedirs(out_dir, exist_ok=True)
    epi_dir = os.path.join(out_dir, "EPI")
    if os.path.isdir(epi_dir) and len(os.listdir(epi_dir)) > 0:
        print(f"EPI directory already exists and is non-empty: {epi_dir}")
        print("Skipping download instructions. Inspecting layout instead.")
        return True

    print("=" * 60)
    print("GUE+ EPI Download Instructions")
    print("=" * 60)
    print()
    print("DNABERT-2 distributes the GUE+ benchmark via Google Drive.")
    print("Automatic download usually requires `gdown`:")
    print("  pip install gdown")
    print("  gdown --folder <GDRIVE_FOLDER_ID> -O", out_dir)
    print()
    print("The Google Drive folder ID is listed in the DNABERT-2 README:")
    print(f"  {GUE_PLUS_GDRIVE_README}")
    print()
    print("Alternatively, download from Zenodo:")
    print(f"  {GUE_PLUS_ZENODO}")
    print()
    print("After downloading, place the EPI/ folder under:")
    print(f"  {out_dir}/EPI/")
    print()
    print("Then re-run this script with --inspect-only to verify the layout.")
    return False


def main():
    args = parse_args()
    if args.inspect_only:
        inspect_epi_layout(args.out_dir)
        return

    ready = print_download_instructions(args.out_dir)
    if ready or os.path.isdir(os.path.join(args.out_dir, "EPI")):
        inspect_epi_layout(args.out_dir)


if __name__ == "__main__":
    main()
