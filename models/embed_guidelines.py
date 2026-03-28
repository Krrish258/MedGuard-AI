#!/usr/bin/env python3
"""
MedGuard-AI — Clinical Guidelines Embedder
Downloads PubMedBERT locally (one-time) and embeds the clinical
guidelines corpus for semantic retrieval at inference time.

Usage:
    python models/embed_guidelines.py

Output:
    models/guideline_embeddings.pkl   → {text: embedding} index
    models/guideline_index.json       → {disease: guideline_text} lookup
"""

import json
from pathlib import Path

import joblib
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
OUT    = ROOT / "models"

GUIDELINES_FILE = KB_DIR / "treatment_guidelines.json"
MODEL_NAME      = "neuml/pubmedbert-base-embeddings"


def build_corpus(guidelines: dict) -> dict[str, str]:
    """
    Construct a disease → guideline text corpus from treatment_guidelines.json.
    Each entry is a concise clinical summary for embedding.
    """
    corpus: dict[str, str] = {}
    for disease, data in guidelines.items():
        if disease.startswith("_"):
            continue
        text = (
            f"Disease: {disease}. "
            f"ICD-10: {data.get('icd10', 'N/A')}. "
            f"First-line treatment: {data.get('first_line_drug', 'N/A')}. "
            f"Dose: {data.get('dose', 'N/A')}. "
            f"Drug class: {data.get('drug_class', 'N/A')}. "
            f"Reasoning: {data.get('reasoning', '')}. "
            f"Evidence level: {data.get('evidence_level', 'N/A')}. "
            f"Alternative: {data.get('alternative', 'N/A')}. "
            f"Monitoring: {data.get('monitoring', 'N/A')}."
        )
        corpus[disease] = text
    return corpus


def embed_corpus(corpus: dict[str, str]) -> dict[str, np.ndarray]:
    """
    Embed each guideline text using PubMedBERT sentence transformer.
    Downloads model on first run, then uses local cache.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers torch")
        raise

    print(f"[INFO] Loading model: {MODEL_NAME}")
    print("[INFO] First run will download ~400MB model to ~/.cache/huggingface/")
    model = SentenceTransformer(MODEL_NAME)

    texts   = list(corpus.values())
    keys    = list(corpus.keys())

    print(f"[INFO] Embedding {len(texts)} clinical guidelines...")
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_numpy=True, batch_size=16)

    return {disease: embeddings[i] for i, disease in enumerate(keys)}


def main() -> None:
    print("=" * 60)
    print("  MedGuard-AI — Clinical Guidelines Embedder")
    print("=" * 60)

    with open(GUIDELINES_FILE) as f:
        guidelines = json.load(f)

    corpus = build_corpus(guidelines)
    print(f"[OK] Built corpus: {len(corpus)} disease guidelines")

    embedding_index = embed_corpus(corpus)

    # Save embeddings index
    embeddings_path = OUT / "guideline_embeddings.pkl"
    joblib.dump(embedding_index, embeddings_path, compress=3)
    print(f"[OK] Embeddings saved → {embeddings_path}")

    # Save text index for display purposes
    index_path = OUT / "guideline_index.json"
    with open(index_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"[OK] Guideline text index saved → {index_path}")

    print("\n[DONE] Embedding complete. Next step:")
    print("  python cli/main.py --patient schema/example_patients.json --id PT-001")


if __name__ == "__main__":
    main()
