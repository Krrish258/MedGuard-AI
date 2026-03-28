#!/usr/bin/env python3
"""
MedGuard-AI — Ensemble Diagnosis Classifier Trainer
Trains a 3-model majority-vote ensemble on the Kaggle disease-symptom dataset.

Classifiers:
  1. RandomForestClassifier       (robust, feature-importance)
  2. GradientBoostingClassifier   (high accuracy on tabular data)
  3. SVC (probability=True)       (strong decision boundaries)

Outputs:
  models/diagnosis_classifier.pkl  → trained VotingClassifier
  models/label_encoder.pkl         → sklearn LabelEncoder

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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX = PROC_DIR / "feature_matrix.csv"
SYMPTOM_LIST   = PROC_DIR / "symptom_list.json"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
GB_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
)
SVC_PARAMS = dict(
    kernel="rbf",
    C=10.0,
    probability=True,
    class_weight="balanced",
    random_state=42,
)

CONFIDENCE_THRESHOLD = 0.60   # below this → agent considers abstaining
CV_FOLDS             = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def build_ensemble() -> VotingClassifier:
    """Construct the 3-model soft-voting ensemble."""
    rf  = RandomForestClassifier(**RF_PARAMS)
    gb  = GradientBoostingClassifier(**GB_PARAMS)
    svc = SVC(**SVC_PARAMS)

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest",       rf),
            ("gradient_boosting",   gb),
            ("svm",                 svc),
        ],
        voting="soft",          # average class probabilities
        weights=[2, 2, 1],      # RF + GB weighted slightly higher than SVC
    )
    return ensemble


def cross_validate(model, X: np.ndarray, y: np.ndarray) -> None:
    """Run stratified k-fold cross-validation and print summary."""
    print(f"\n[INFO] Running {CV_FOLDS}-fold stratified cross-validation...")
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold:    {[round(s, 4) for s in scores]}")


def train_and_evaluate(
    model: VotingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
) -> VotingClassifier:
    """Train on 80%, evaluate on 20%, print full report."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"\n[INFO] Training ensemble on {len(X_train)} samples...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[OK]  Training complete in {elapsed:.1f}s")

    # ── Evaluation ───────────────────────────────────────────
    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)
    confidence = y_proba.max(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Test-set Results ──────────────────────────────────────")
    print(f"  Accuracy           : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Mean Confidence    : {confidence.mean():.4f}")
    print(f"  Low-conf samples   : {(confidence < CONFIDENCE_THRESHOLD).sum()} "
          f"/ {len(y_test)} "
          f"(< {CONFIDENCE_THRESHOLD:.0%} threshold)")

    print("\n── Per-Class Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion matrix heatmap ─────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(cm, le.classes_)

    # ── Feature importance (from RF sub-model) ───────────────
    _print_top_features(model, X.shape[1])

    return model


def _save_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Save confusion matrix plot to models/confusion_matrix.png."""
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
        linewidths=0.4,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("MedGuard-AI — Ensemble Diagnosis Classifier\nConfusion Matrix", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    out = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved → {out}")


def _print_top_features(model: VotingClassifier, n_features: int, top_n: int = 20) -> None:
    """Print top-N most important symptoms from the RF sub-model."""
    rf_model    = model.named_estimators_["random_forest"]
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    # Re-load symptom names for display
    with open(SYMPTOM_LIST) as f:
        symptom_names = json.load(f)

    print(f"\n── Top {top_n} Diagnostic Symptom Features (RF Importance) ──")
    for rank, idx in enumerate(indices, 1):
        name = symptom_names[idx] if idx < len(symptom_names) else f"feature_{idx}"
        print(f"  {rank:>2}. {name:<40} {importances[idx]:.4f}")


def save_model(model: VotingClassifier, le: LabelEncoder) -> None:
    """Persist the trained ensemble and label encoder."""
    model_path = MODELS_DIR / "diagnosis_classifier.pkl"
    le_path    = MODELS_DIR / "label_encoder.pkl"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(le,    le_path,    compress=3)

    print(f"\n[OK] Model saved    → {model_path}")
    print(f"[OK] Encoder saved  → {le_path}")


def main() -> None:
    print("=" * 60)
    print("  MedGuard-AI — Ensemble Classifier Training")
    print("=" * 60)

    # 1. Load processed data
    X, y, symptom_names, le = load_data()

    # 2. Build the 3-model ensemble
    model = build_ensemble()
    print(f"[INFO] Built VotingClassifier with estimators: {[n for n,_ in model.estimators]}")

    # 3. Cross-validate (uses full dataset to assess generalisation)
    cross_validate(model, X, y)

    # 4. Final train on 80% split, evaluate on 20%
    model = train_and_evaluate(model, X, y, le)

    # 5. Retrain on FULL dataset before saving (best for deployment)
    print("\n[INFO] Retraining on full dataset for deployment...")
    model.fit(X, y)
    print("[OK]  Full-dataset retraining complete.")

    # 6. Save artifacts
    save_model(model, le)

    print("\n[DONE] Model training complete. Next step:")
    print("  python knowledge_base/build_knowledge_graph.py")


if __name__ == "__main__":
    main()
