#!/usr/bin/env python3
"""
MedGuard-AI — Logistic Regression Classifier Trainer
Trains a Logistic Regression model on the dataset.

Outputs:
  models/diagnosis_classifier.pkl
  models/label_encoder.pkl

Usage:
    python models/train_classifier.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX = PROC_DIR / "feature_matrix.csv"
SYMPTOM_LIST   = PROC_DIR / "symptom_list.json"


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """Load processed feature matrix and return X, y, symptom names, encoder."""
    if not FEATURE_MATRIX.exists():
        print("[ERROR] feature_matrix.csv not found. Run: python data/preprocessor.py")
        sys.exit(1)

    df = pd.read_csv(FEATURE_MATRIX)
    with open(SYMPTOM_LIST) as f:
        symptom_names = json.load(f)

    X = df[symptom_names].values.astype(float)
    y_raw = df["prognosis"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    print(f"[OK] Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    return X, y, symptom_names, le


def main() -> None:
    print("=" * 60)
    print("  MedGuard-AI — Logistic Regression Training")
    print("=" * 60)

    # 1. Load data
    X, y, symptom_names, le = load_data()

    # 2. Build Model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    print(f"\n[INFO] Training model on {len(X_train)} samples...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[OK]  Training complete in {elapsed:.1f}s")

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n── Test-set Results ──────────────────────────────────────")
    print(f"  Accuracy           : {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n── Per-Class Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 5. Full Retrain
    print("\n[INFO] Retraining on full dataset for deployment...")
    model.fit(X, y)

    # 6. Save
    model_path = MODELS_DIR / "diagnosis_classifier.pkl"
    le_path    = MODELS_DIR / "label_encoder.pkl"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(le,    le_path,    compress=3)

    print(f"\n[OK] Model saved    → {model_path}")
    print(f"[OK] Encoder saved  → {le_path}")

if __name__ == "__main__":
    main()
