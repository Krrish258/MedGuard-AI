"""
MedGuard-AI — Symptom Matcher
Maps free-text patient symptoms to the canonical 132-symptom vocabulary
used during model training, via fuzzy string matching.
"""

from __future__ import annotations
import json
from pathlib import Path

from thefuzz import process as fuzz_process
import os
import io

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    genai = None

ROOT = Path(__file__).resolve().parent.parent.parent
SYMPTOM_LIST_PATH = ROOT / "data" / "processed" / "symptom_list.json"

# Minimum fuzzy-match score to accept a match (0–100)
MATCH_THRESHOLD = 70


class SymptomMatcher:
    """
    Converts free-text symptom strings into canonical vocabulary entries.
    Uses token_sort_ratio for robust matching (handles word order differences).
    """

    def __init__(self) -> None:
        self.vocabulary: list[str] = self._load_vocabulary()
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.model = None

    def _load_vocabulary(self) -> list[str]:
        if not SYMPTOM_LIST_PATH.exists():
            raise FileNotFoundError(
                f"Symptom vocabulary not found at {SYMPTOM_LIST_PATH}.\n"
                "Please run: python data/preprocessor.py"
            )
        with open(SYMPTOM_LIST_PATH) as f:
            return json.load(f)

    def match(self, raw_symptoms: list[str]) -> tuple[list[str], list[dict]]:
        """
        Match each raw symptom string to the closest canonical vocabulary entry.

        Args:
            raw_symptoms: list of free-text symptom strings

        Returns:
            matched_symptoms : list of canonical symptom names (in vocabulary)
            match_report     : list of {raw, matched, score} dicts for transparency
        """
        matched: list[str] = []
        report: list[dict]  = []
        
        # If the input is a single long natural-language prompt, use Semantic Extraction
        if len(raw_symptoms) == 1 and len(raw_symptoms[0]) > 20 and self.model:
            prompt = raw_symptoms[0]
            vocab_str = ", ".join(self.vocabulary)
            llm_prompt = f"Analyze the following patient prompt and extract the exact symptoms present from this allowed vocabulary list ONLY:\n[{vocab_str}]\n\nPatient Prompt: '{prompt}'\n\nOutput only the comma-separated extracted symptoms, nothing else."
            try:
                response = self.model.generate_content(llm_prompt)
                extracted = [s.strip().lower().replace(" ", "_") for s in response.text.split(",") if s.strip()]
                # Re-validate against vocabulary
                valid_extracted = [s for s in extracted if s in self.vocabulary]
                if valid_extracted:
                    for s in valid_extracted:
                        matched.append(s)
                        report.append({"raw": prompt[:30]+"...", "matched": s, "score": 99, "method": "Gemini LLM Semantic"})
                    return matched, report
            except Exception as e:
                print(f"[WARN] Gemini extraction failed, falling back to fuzzy match: {e}")

        for raw in raw_symptoms:
            normalised = raw.strip().lower().replace(" ", "_")

            # 1. Exact match first (fastest path)
            if normalised in self.vocabulary:
                matched.append(normalised)
                report.append({"raw": raw, "matched": normalised, "score": 100, "method": "exact"})
                continue

            # 2. Fuzzy match using token_sort_ratio
            result = fuzz_process.extractOne(
                normalised,
                self.vocabulary,
                scorer=fuzz_process.fuzz.token_sort_ratio,
            )

            if result and result[1] >= MATCH_THRESHOLD:
                best_match, score = result[0], result[1]
                matched.append(best_match)
                report.append({"raw": raw, "matched": best_match, "score": score, "method": "fuzzy"})
            else:
                # No confident match — log as unmatched but don't add to feature vector
                report.append({
                    "raw": raw,
                    "matched": None,
                    "score": result[1] if result else 0,
                    "method": "unmatched",
                })

        return matched, report

    def to_feature_vector(self, matched_symptoms: list[str]) -> list[int]:
        """
        Convert matched symptom list to a binary feature vector
        aligned with the training vocabulary.

        Args:
            matched_symptoms: canonical symptom names

        Returns:
            list of 0/1 integers, length = len(vocabulary)
        """
        symptom_set = set(matched_symptoms)
        return [1 if sym in symptom_set else 0 for sym in self.vocabulary]
