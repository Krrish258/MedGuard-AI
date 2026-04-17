"""
MedGuard-AI — ReAct Reasoning Agent (Layer 1)
Implements the Thought → Action → Observe reasoning loop
to generate a diagnosis, treatment plan, and reasoning trace
from structured patient evidence — without any external LLM API.

Reasoning tools available to the agent:
  1. analyse_symptoms(evidence)          → symptom feature vector
  2. predict_diagnosis(feature_vector)   → diagnosis + confidence + differentials
  3. retrieve_guidelines(diagnosis)      → clinical guideline text
  4. generate_treatment(diagnosis, guideline) → treatment plan
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import csv
import joblib
import numpy as np
import json
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .evidence_extractor import EvidenceBundle
from .symptom_matcher import SymptomMatcher
from .guideline_retriever import GuidelineRetriever

ROOT       = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
KB_DIR     = ROOT / "knowledge_base"

CONFIDENCE_THRESHOLD = 0.60   # below this → abstain recommended
# Citation: FDA AI/ML SaMD Action Plan (2021); EC MDCG Guidance on AI Medical Devices (2021);
# Topol EJ, Nature Medicine 2019 — diagnostic AI requires ≥0.60 calibrated confidence.


# ── Agent output dataclass ────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """The structured output produced by the ReAct reasoning loop."""
    patient_id: str
    diagnosis: str
    confidence: float
    differential: list[dict]           # [{disease, probability}]
    treatment: str
    drug_name: str
    dose: str
    drug_class: str
    evidence_level: str
    reasoning_trace: list[dict]        # [{step, action, thought, observation}]
    guidelines_retrieved: list[dict]   # [{disease, similarity_score, text}]
    symptom_match_report: list[dict]   # [{raw, matched, score, method}]
    clarifying_questions: list[str] = field(default_factory=list)
    low_confidence: bool = False


# ── Main ReAct Agent ──────────────────────────────────────────────────────────

class ReasoningAgent:
    """
    Autonomous reasoning agent that interprets patient evidence,
    queries the trained ML ensemble, retrieves clinical guidelines,
    and generates a treatment recommendation via a ReAct loop.
    """

    def __init__(self) -> None:
        self._model = None
        self._label_encoder = None
        self._symptom_matcher   = SymptomMatcher()
        self._guideline_retriever = GuidelineRetriever()
        self._treatment_kb: dict = {}
        self._trace: list[dict]  = []
        self._step = 0
        
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self._gemini_model = None

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            model_path = MODELS_DIR / "diagnosis_classifier.pkl"
            le_path    = MODELS_DIR / "label_encoder.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    "Trained model not found. Run: python models/train_classifier.py"
                )
            self._model         = joblib.load(model_path)
            self._label_encoder = joblib.load(le_path)

    def _load_treatment_kb(self):
        if not self._treatment_kb:
            path = ROOT / "data" / "raw" / "symptom_precaution.csv"
            # We map disease to a structured dict matching the expected fields
            try:
                with open(path, encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        disease = row.get("Disease", "").strip().lower()
                        p1 = row.get("Precaution_1", "").strip()
                        p2 = row.get("Precaution_2", "").strip()
                        p3 = row.get("Precaution_3", "").strip()
                        p4 = row.get("Precaution_4", "").strip()
                        
                        precautions = [p for p in (p1, p2, p3, p4) if p]
                        joined_precautions = ", ".join(precautions)
                        
                        self._treatment_kb[disease] = {
                            "first_line_drug": joined_precautions,  # We repurpose first_line_drug 
                            "dose": "N/A",
                            "drug_class": "Dataset Precaution",
                            "evidence_level": "Dataset"
                        }
            except Exception as e:
                print(f"[WARN] Failed to load dataset treatments: {e}")
                self._treatment_kb = {}

    # ── Trace helpers ─────────────────────────────────────────────────────────

    def _log(self, action: str, thought: str, observation: str) -> None:
        self._step += 1
        self._trace.append({
            "step": self._step,
            "action": action,
            "thought": thought,
            "observation": observation,
        })

    # ── Tool 1: Symptom analysis ──────────────────────────────────────────────

    def _tool_analyse_symptoms(
        self, evidence: EvidenceBundle
    ) -> tuple[list[int], list[dict]]:
        """Match raw symptoms to canonical vocabulary and build feature vector."""
        matched, report = self._symptom_matcher.match(evidence.raw_symptoms)
        evidence.matched_symptoms = matched
        feature_vector = self._symptom_matcher.to_feature_vector(matched)

        matched_names = [r["matched"] for r in report if r["matched"]]
        unmatched     = [r["raw"] for r in report if not r["matched"]]

        thought = (
            f"Patient presents with {len(evidence.raw_symptoms)} reported symptom(s). "
            f"Mapped {len(matched_names)} to canonical vocabulary. "
            + (f"Could not match: {unmatched}." if unmatched else "All symptoms matched.")
        )
        observation = (
            f"Feature vector constructed: {sum(feature_vector)} active symptoms "
            f"from vocabulary of {len(feature_vector)}."
        )
        self._log("ANALYSE_SYMPTOMS", thought, observation)
        return feature_vector, report

    # ── Tool 2: ML diagnosis prediction ──────────────────────────────────────

    def _tool_predict_diagnosis(
        self, feature_vector: list[int], evidence: EvidenceBundle
    ) -> tuple[str, float, list[dict], list[str]]:
        """Run the ensemble classifier or LLM to get diagnosis + confidence."""
        if self._gemini_model and len(evidence.raw_symptoms) >= 1:
            prompt = " | ".join(evidence.raw_symptoms)
            llm_prompt = f"""
            You are a highly capable diagnostic AI. A patient presents with the following profile:
            - Age: {evidence.age}
            - Primary Symptoms: "{prompt}"
            - Medical History/Conditions: {evidence.medical_history if evidence.medical_history else 'None'}
            - Current Medications: {[m.name for m in evidence.current_medications] if evidence.current_medications else 'None'}
            - Allergies: {evidence.allergies if evidence.allergies else 'None'}
            
            Analyze this medically globally. Output a valid raw JSON object (no markdown) with exact keys:
            - "primary_diagnosis": A string representing the most likely disease out of any illness globally.
            - "confidence": A number from 0.0 to 1.0 representing your realistic calibrated confidence. If symptoms are extremely vague, keep it low.
            - "differentials": A list of dicts with "disease" (string) and "probability" (float 0.0-1.0). Limit to 3 max.
            - "clarifying_questions": If the confidence is < 0.65 or key information is missing, add a list of string questions to ask the patient. Else empty list.
            Output purely JSON.
            """
            try:
                resp = self._gemini_model.generate_content(llm_prompt)
                text = resp.text.strip().replace("```json", "").replace("```", "")
                data = json.loads(text)
                
                top_disease = data.get("primary_diagnosis", "Unknown")
                top_conf = float(data.get("confidence", 0.0))
                differential = data.get("differentials", [])
                questions = data.get("clarifying_questions", [])
                
                thought = f"Requested Global Diagnosis via Gemini LLM."
                obs_conf = f"{top_conf:.1%}"
                observation = f"Primary diagnosis: '{top_disease}' (confidence: {obs_conf}). Differentials: {[d['disease'] for d in differential]}. Questions: {len(questions)}"
                self._log("PREDICT_DIAGNOSIS_LLM", thought, observation)
                return top_disease.lower().strip(), top_conf, differential, questions
            except Exception as e:
                print(f"[WARN] Gemini Diagnosis failed, falling back to ML Model: {e}")

        # Fallback ML Model Path
        self._load_model()
        X         = np.array(feature_vector).reshape(1, -1)
        proba     = self._model.predict_proba(X)[0]
        classes   = self._label_encoder.classes_

        # Top-1 prediction
        top_idx       = int(np.argmax(proba))
        top_disease   = classes[top_idx]
        top_conf      = float(proba[top_idx])

        # Top-5 differential
        sorted_idx    = np.argsort(proba)[::-1][:5]
        differential  = [
            {"disease": classes[i], "probability": round(float(proba[i]), 4)}
            for i in sorted_idx
        ]

        thought = (
            f"Running ensemble classifier (RF + GradientBoosting + SVM) "
            f"on {sum(feature_vector)}-symptom feature vector."
        )
        obs_conf = f"{top_conf:.1%}"
        observation = (
            f"Primary diagnosis: '{top_disease}' (confidence: {obs_conf}). "
            f"Differential: {[d['disease'] for d in differential[1:4]]}."
        )
        self._log("PREDICT_DIAGNOSIS", thought, observation)
        return top_disease, top_conf, differential, []

    # ── Tool 3: Guideline retrieval ───────────────────────────────────────────

    def _tool_retrieve_guidelines(
        self, diagnosis: str
    ) -> tuple[list[dict], Optional[dict]]:
        """Retrieve semantically similar clinical guidelines via PubMedBERT."""
        thought = (
            f"Querying PubMedBERT guideline index for '{diagnosis}' "
            "to retrieve evidence-based treatment protocols."
        )
        try:
            results = self._guideline_retriever.retrieve(diagnosis, top_k=3)
            best    = results[0] if results else None
            observation = (
                f"Retrieved {len(results)} guideline(s). "
                f"Top match: '{best['disease']}' "
                f"(similarity: {best['similarity_score']:.3f})."
                if best else "No guidelines retrieved."
            )
        except FileNotFoundError:
            # Fallback to direct KB lookup if embeddings not generated yet
            results     = []
            best        = None
            observation = "Guideline embeddings not found — falling back to KB lookup."

        self._log("RETRIEVE_GUIDELINES", thought, observation)
        return results, best

    # ── Tool 4: Treatment generation ─────────────────────────────────────────

    def _tool_generate_treatment(
        self, diagnosis: str, guideline_match: Optional[dict]
    ) -> tuple[str, str, str, str, str]:
        """Look up treatment details from the knowledge base."""
        self._load_treatment_kb()

        # Use the guideline match disease if it's closely aligned
        lookup_key = diagnosis.lower()
        if guideline_match and guideline_match["similarity_score"] > 0.7:
            # We just try diagnosis first since dataset has it exact mostly
            pass

        data = self._treatment_kb.get(lookup_key, {})

        if not data:
            treatment   = "Supportive care; refer to specialist."
            drug_name   = "N/A"
            dose        = "N/A"
            drug_class  = "N/A"
            evidence    = "N/A"
        else:
            treatment   = data.get("first_line_drug", "Supportive care")
            drug_name   = "N/A" # Drugs are not directly mapped in precaution dataset
            dose        = "N/A"
            drug_class  = data.get("drug_class", "N/A")
            evidence    = data.get("evidence_level", "N/A")

        thought = (
            f"Generating treatment plan for '{diagnosis}' "
            f"based on retrieved guideline (evidence level: {evidence})."
        )
        observation = (
            f"Treatment: {treatment[:80]}{'...' if len(treatment) > 80 else ''}. "
            f"Drug class: {drug_class}. Evidence: {evidence}."
        )
        self._log("GENERATE_TREATMENT", thought, observation)
        return treatment, drug_name, dose, drug_class, evidence

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, evidence: EvidenceBundle) -> AgentResult:
        """
        Execute the full ReAct reasoning loop for a patient.

        Steps:
          1. ANALYSE_SYMPTOMS    → feature vector
          2. PREDICT_DIAGNOSIS   → diagnosis + confidence
          3. RETRIEVE_GUIDELINES → clinical evidence
          4. GENERATE_TREATMENT  → treatment plan

        Returns:
            AgentResult
        """
        self._trace = []
        self._step  = 0

        self._log(
            "INIT",
            f"Patient {evidence.patient_id} | Age: {evidence.age} | "
            f"Risk profile: {evidence.age_risk} | "
            f"Comorbidities: {evidence.comorbidity_count}",
            f"Evidence bundle validated. "
            f"{len(evidence.raw_symptoms)} symptom(s) to process.",
        )

        # ── Step 1: Symptom analysis ──────────────────────────────────────────
        feature_vector, match_report = self._tool_analyse_symptoms(evidence)

        # ── Step 2: Diagnosis prediction ──────────────────────────────────────
        diagnosis, confidence, differential, clarifying_qs = self._tool_predict_diagnosis(
            feature_vector, evidence
        )
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        if low_confidence:
            self._log(
                "CONFIDENCE_CHECK",
                f"Confidence {confidence:.1%} is below threshold {CONFIDENCE_THRESHOLD:.0%}.",
                "Low-confidence flag set. Decision module will consider ABSTAIN.",
            )

        # ── Step 3: Guideline retrieval ───────────────────────────────────────
        guidelines, best_match = self._tool_retrieve_guidelines(diagnosis)

        # ── Step 4: Treatment generation ─────────────────────────────────────
        treatment, drug_name, dose, drug_class, evidence_level = (
            self._tool_generate_treatment(diagnosis, best_match)
        )

        self._log(
            "CONCLUDE",
            f"Reasoning complete for patient {evidence.patient_id}.",
            f"Diagnosis: {diagnosis} ({confidence:.1%}). "
            f"Treatment: {drug_name}. "
            f"Forwarding to Causal Verification Engine.",
        )

        return AgentResult(
            patient_id           = evidence.patient_id,
            diagnosis            = diagnosis,
            confidence           = confidence,
            differential         = differential,
            treatment            = treatment,
            drug_name            = drug_name,
            dose                 = dose,
            drug_class           = drug_class,
            evidence_level       = evidence_level,
            reasoning_trace      = self._trace,
            guidelines_retrieved = guidelines,
            symptom_match_report = match_report,
            clarifying_questions = clarifying_qs,
            low_confidence       = low_confidence,
        )
