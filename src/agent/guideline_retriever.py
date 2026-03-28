"""
MedGuard-AI — Guideline Retriever
Performs semantic search over PubMedBERT-embedded clinical guidelines
to retrieve the most relevant evidence for a given diagnosis.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

EMBEDDINGS_PATH = ROOT / "models" / "guideline_embeddings.pkl"
INDEX_PATH      = ROOT / "models" / "guideline_index.json"

EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"


class GuidelineRetriever:
    """
    Retrieves the top-k most semantically similar clinical guidelines
    for a given query (diagnosis name or symptom description).
    Uses cosine similarity over PubMedBERT embeddings.
    """

    def __init__(self) -> None:
        self._embedding_index: Optional[dict[str, np.ndarray]] = None
        self._text_index: Optional[dict[str, str]] = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._embedding_index is not None:
            return
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Guideline embeddings not found. "
                "Run: python models/embed_guidelines.py"
            )
        import json
        self._embedding_index = joblib.load(EMBEDDINGS_PATH)
        with open(INDEX_PATH) as f:
            self._text_index = json.load(f)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
        return self._model

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most relevant clinical guidelines.

        Args:
            query  : diagnosis name or symptom cluster description
            top_k  : number of results to return

        Returns:
            list of dicts: {disease, guideline_text, similarity_score}
        """
        self._ensure_loaded()

        # Embed the query
        model     = self._get_model()
        query_emb = model.encode(query, convert_to_numpy=True)

        # Compute cosine similarity vs all guideline embeddings
        scores: list[tuple[str, float]] = []
        for disease, emb in self._embedding_index.items():
            score = self._cosine_similarity(query_emb, emb)
            scores.append((disease, score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        for disease, score in top:
            results.append({
                "disease"         : disease,
                "guideline_text"  : self._text_index.get(disease, ""),
                "similarity_score": round(score, 4),
            })
        return results

    def get_guideline(self, disease: str) -> Optional[str]:
        """
        Direct lookup: retrieve guideline text for a specific disease.

        Args:
            disease: exact disease name (lowercased)

        Returns:
            guideline text or None if not found
        """
        self._ensure_loaded()
        return self._text_index.get(disease.lower())
