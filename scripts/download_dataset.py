#!/usr/bin/env python3
"""
MedGuard-AI — Kaggle Dataset Downloader
Downloads the Disease Symptom Prediction dataset and places
the CSVs into data/raw/.

Usage:
    python scripts/download_dataset.py

Requirements:
    - A valid ~/.kaggle/kaggle.json API token
      (Get yours from: https://www.kaggle.com/settings → API → Create New Token)
    - pip install kaggle
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
KAGGLE_KEY = Path.home() / ".kaggle" / "kaggle.json"

DATASET_SLUG = "itachi9604/disease-symptom-description-dataset"
EXPECTED_FILES = [
    "Training.csv",
    "Testing.csv",
    "Symptom-severity.csv",
    "symptom_precaution.csv",
]


def check_kaggle_token() -> None:
    """Verify that kaggle.json exists and is readable."""
    if not KAGGLE_KEY.exists():
        print("[ERROR] Kaggle API token not found.")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. Run:  chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print(f"[OK] Kaggle token found at {KAGGLE_KEY}")


def download_dataset() -> None:
    """Download and unzip the dataset to data/raw/ using the Kaggle Python API."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((RAW_DIR / f).exists() for f in EXPECTED_FILES):
        print("[OK] Dataset already present in data/raw/ — skipping download.")
        return

    print(f"[INFO] Downloading dataset: {DATASET_SLUG}")

    try:
        import kaggle.api as kaggle_api
        api = kaggle_api
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle API authentication failed: {e}")
        print("  Make sure ~/.kaggle/kaggle.json exists and has correct credentials.")
        sys.exit(1)

    print("[INFO] Authenticated with Kaggle API. Starting download...")
    try:
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(RAW_DIR),
            unzip=True,
            force=True,
            quiet=False,
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

    print(f"[OK] Downloaded and extracted to {RAW_DIR}")


def verify_files() -> None:
    """Confirm all expected CSVs are present."""
    print("\n[INFO] Verifying dataset files...")
    all_ok = True
    for fname in EXPECTED_FILES:
        path = RAW_DIR / fname
        if path.exists():
            size_kb = path.stat().st_size // 1024
            print(f"  ✅  {fname:<35} ({size_kb} KB)")
        else:
            print(f"  ❌  {fname:<35} MISSING")
            all_ok = False

    if not all_ok:
        print("\n[WARNING] Some files are missing. The dataset structure may have changed.")
        print("  Please check: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        sys.exit(1)
    else:
        print("\n[OK] All dataset files verified. Ready for preprocessing.")


if __name__ == "__main__":
    print("=" * 60)
    print("  MedGuard-AI — Kaggle Dataset Downloader")
    print("=" * 60)
    check_kaggle_token()
    download_dataset()
    verify_files()
