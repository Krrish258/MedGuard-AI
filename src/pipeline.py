"""
MedGuard-AI — Pipeline Orchestrator
Connects all three layers into a single end-to-end function.

Usage:
    from src.pipeline import MedGuardPipeline
    pipeline = MedGuardPipeline()
    result   = pipeline.run(patient_dict)
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any

from .agent.evidence_extractor import EvidenceBundle, EvidenceExtractor
from .agent.reasoning_agent import AgentResult, ReasoningAgent
from .verifier.causal_verifier import CausalVerifier, VerificationResult
from .decision_module import DecisionModule, FinalDecision


class MedGuardPipeline:
    """
    End-to-end MedGuard-AI pipeline.

      patient_dict
          ↓  EvidenceExtractor
      EvidenceBundle
          ↓  ReasoningAgent (Layer 1)
      AgentResult
          ↓  CausalVerifier (Layer 2)
      VerificationResult
          ↓  DecisionModule (Layer 3)
      FinalDecision
    """

    def __init__(self) -> None:
        self._extractor  = EvidenceExtractor()
        self._agent      = ReasoningAgent()
        self._verifier   = CausalVerifier()
        self._decision   = DecisionModule()

    def run(self, patient_data: dict[str, Any]) -> dict:
        """
        Run the full pipeline on a patient dict.

        Args:
            patient_data: dict conforming to patient_schema.json

        Returns:
            Final output dict (JSON-serialisable)
        """
        # Layer 0: Extract & validate evidence
        evidence: EvidenceBundle = self._extractor.extract(patient_data)

        # Layer 1: Reasoning Agent
        agent_result: AgentResult = self._agent.run(evidence)

        # Layer 2: Causal Verification
        verification: VerificationResult = self._verifier.verify(evidence, agent_result)

        # Layer 3: Decision
        decision: FinalDecision = self._decision.decide(agent_result, verification)

        # ── Assemble final output ─────────────────────────────────────────────
        return {
            "patient_id"     : evidence.patient_id,
            "age"            : evidence.age,
            "age_risk"       : evidence.age_risk,

            # Agent output
            "diagnosis"      : agent_result.diagnosis,
            "confidence"     : round(agent_result.confidence, 4),
            "differential"   : agent_result.differential[:5],
            "treatment"      : agent_result.treatment,
            "drug_name"      : agent_result.drug_name,
            "dose"           : agent_result.dose,
            "drug_class"     : agent_result.drug_class,
            "evidence_level" : agent_result.evidence_level,
            "clarifying_questions": agent_result.clarifying_questions,

            # Verification output
            "safety"         : {
                "score"           : verification.safety_score,
                "is_safe"         : verification.is_safe,
                "penalties"       : verification.penalties,
                "risk_label"      : verification.risk_label,
                "risk_factors"    : verification.risk_factors,
                "allergy_issues"  : verification.allergy_issues,
                "interaction_issues": verification.interaction_issues,
                "history_issues"  : verification.history_issues,
            },

            # Decision
            "decision"       : decision.decision,
            "message"        : decision.message,

            # Transparency
            "reasoning_trace": agent_result.reasoning_trace,
            "symptom_matching": agent_result.symptom_match_report,
            "guidelines_retrieved": [
                {"disease": g["disease"], "similarity": g["similarity_score"]}
                for g in agent_result.guidelines_retrieved
            ],
        }
