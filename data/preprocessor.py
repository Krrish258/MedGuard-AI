#!/usr/bin/env python3
"""
MedGuard-AI — Data Preprocessor
Handles the Kaggle disease-symptom dataset (updated format):
  dataset.csv  →  Disease | Symptom_1 | ... | Symptom_17  (long format)

Converts to binary feature matrix and produces:
  data/processed/symptom_list.json    → canonical symptom vocabulary
  data/processed/feature_matrix.csv  → binary matrix (one-hot symptoms)
  data/processed/label_encoder.json  → disease name ↔ index

Usage:
    python data/preprocessor.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_name(s: str) -> str:
    """Normalise a symptom or disease name to lowercase snake_case."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def load_dataset() -> pd.DataFrame:
    """Load dataset.csv — new format: Disease + Symptom_1..Symptom_17."""
    path = RAW_DIR / "dataset.csv"
    if not path.exists():
        print("[ERROR] dataset.csv not found in data/raw/")
        print("  Run: python scripts/download_dataset.py")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[OK] Loaded dataset.csv → {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def build_binary_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert long-format symptom columns (Symptom_1..Symptom_17) into
    a wide-format binary feature matrix.

    Input:  Disease | Symptom_1 | Symptom_2 | ... | Symptom_17
    Output: symptom_a | symptom_b | ... | symptom_n | prognosis
    """
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

    # Gather all unique symptom names
    all_symptoms: set[str] = set()
    for col in symptom_cols:
        vals = df[col].dropna().apply(clean_name)
        all_symptoms.update(vals)
    all_symptoms.discard("")
    symptom_vocab = sorted(list(all_symptoms))
    print(f"[INFO] Found {len(symptom_vocab)} unique symptoms in dataset")

    # Build binary matrix
    rows = []
    for _, row in df.iterrows():
        disease = clean_name(str(row["Disease"]))
        present = set()
        for col in symptom_cols:
            val = row[col]
            if pd.notna(val):
                cname = clean_name(str(val))
                if cname:
                    present.add(cname)
        binary_row = {sym: (1 if sym in present else 0) for sym in symptom_vocab}
        binary_row["prognosis"] = disease
        rows.append(binary_row)

    matrix = pd.DataFrame(rows)
    print(f"[OK] Built binary matrix: {matrix.shape[0]} rows × {len(symptom_vocab)} symptom features")
    return matrix, symptom_vocab

def load_severity() -> dict[str, int]:
    """Load symptom severity weights from Symptom-severity.csv."""
    path = RAW_DIR / "Symptom-severity.csv"
    if not path.exists():
        print("[WARN] Symptom-severity.csv not found — skipping severity weights.")
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    severity = {}
    for _, row in df.iterrows():
        sym = clean_name(str(row.get("Symptom", "")))
        try:
            severity[sym] = int(row.get("weight", 1))
        except (ValueError, TypeError):
            pass
    sev_path = PROC_DIR / "symptom_severity.json"
    with open(sev_path, "w") as f:
        json.dump(severity, f, indent=2)
    print(f"[OK] Saved severity weights ({len(severity)} entries) → {sev_path}")
    return severity

def save_artifacts(matrix: pd.DataFrame, symptom_vocab: list[str]) -> None:
    """Save feature matrix, symptom vocabulary, and label encoder."""
    
    # Label Encodings
    le = LabelEncoder()
    matrix["label"] = le.fit_transform(matrix["prognosis"])

    # 1. Feature matrix
    matrix_path = PROC_DIR / "feature_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    print(f"[OK] Feature matrix → {matrix_path}  ({len(matrix)} rows)")

    # 2. Symptom vocabulary
    vocab_path = PROC_DIR / "symptom_list.json"
    with open(vocab_path, "w") as f:
        json.dump(symptom_vocab, f, indent=2)
    print(f"[OK] Symptom vocabulary ({len(symptom_vocab)} symptoms) → {vocab_path}")

    # 3. Label encoder (disease name ↔ index)
    label_map = {int(i): name for i, name in enumerate(le.classes_)}
    label_path = PROC_DIR / "label_encoder.json"
    with open(label_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[OK] Label encoder ({len(label_map)} diseases) → {label_path}")

    # 4. Disease distribution
    print("\n── Disease Class Distribution ──────────────────────────────")
    counts = matrix["prognosis"].value_counts()
    for disease, count in counts.items():
        print(f"  {disease:<50} {count:>4} records")

def main() -> None:
    print("=" * 60)
    print("  MedGuard-AI — Data Preprocessor")
    print("=" * 60)

    # 1. Load
    df = load_dataset()

    # 2. Convert long → binary matrix
    matrix, symptom_vocab = build_binary_matrix(df)

    # 3. Severity weights (optional)
    load_severity()

    # 4. Save all artifacts
    print("\n── Saving processed artifacts ──────────────────────────────")
    save_artifacts(matrix, symptom_vocab)

    print("\n[DONE] Preprocessing complete. Next step:")
    print("  python models/train_classifier.py")

if __name__ == "__main__":
    main()
