"""
MedGuard-AI — Causal Verification Engine (Layer 2)
Aggregates all 4 safety checks into a weighted safety score (0–100)
and produces a structured VerificationResult.

Safety Score Formula:
  score = 100
        − allergy_penalty        (0–60)   [CRITICAL: 45, MAJOR: 30, MODERATE: 12]
        − interaction_penalty    (0–60)   [CRITICAL: 40, MAJOR: 25, MODERATE: 10]
        − history_penalty        (0–60)   [CRITICAL: 45, MAJOR: 30, MODERATE: 12]
        − risk_penalty           (0–20)   [age + comorbidity burden]
  score = max(0, min(100, score))
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ..agent.evidence_extractor import EvidenceBundle
from ..agent.reasoning_agent import AgentResult
from .contraindication_checker import ContraindicationChecker
from .interaction_checker import InteractionChecker
from .risk_scorer import RiskScorer


@dataclass
class VerificationResult:
    """Structured output from the Causal Verification Engine."""
    safety_score         : int                    # 0–100
    is_safe              : bool                   # score >= 50
    allergy_issues       : list[dict] = field(default_factory=list)
    interaction_issues   : list[dict] = field(default_factory=list)
    history_issues       : list[dict] = field(default_factory=list)
    risk_factors         : list[str]  = field(default_factory=list)
    risk_label           : str        = "LOW"
    all_issues           : list[dict] = field(default_factory=list)
    penalties            : dict       = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return any(i.get("severity") == "CRITICAL" for i in self.all_issues)

    @property
    def issue_count(self) -> int:
        return len(self.all_issues)


class CausalVerifier:
    """
    Independent verification engine that evaluates agent recommendations
    against structured medical safety rules.
    """

    def __init__(self) -> None:
        self._interaction_checker   = InteractionChecker()
        self._contraindication_checker = ContraindicationChecker()
        self._risk_scorer           = RiskScorer()

    def verify(
        self,
        evidence: EvidenceBundle,
        agent_result: AgentResult,
    ) -> VerificationResult:
        """
        Run all 4 verification checks and compute safety score.

        Args:
            evidence     : EvidenceBundle from the patient
            agent_result : AgentResult from the reasoning agent

        Returns:
            VerificationResult with safety_score, issues, and penalties
        """
        proposed_drug = agent_result.drug_name
        current_meds  = evidence.medication_names
        allergies     = evidence.allergies
        history       = evidence.medical_history

        # ── Check 1: Allergy conflicts ────────────────────────────────────────
        allergy_penalty, allergy_issues = self._contraindication_checker.check_allergies(
            proposed_drug, allergies
        )

        # ── Check 2: Drug-drug interactions ───────────────────────────────────
        interaction_penalty, interaction_issues = self._interaction_checker.check(
            proposed_drug, current_meds
        )

        # ── Check 3: Medical history contraindications ────────────────────────
        history_penalty, history_issues = self._contraindication_checker.check_history(
            proposed_drug, history
        )

        # ── Check 4: Outcome risk assessment ─────────────────────────────────
        risk_penalty, risk_label, risk_factors = self._risk_scorer.score(
            age=evidence.age,
            medical_history=history,
            confidence=agent_result.confidence,
        )

        # ── Aggregate safety score ────────────────────────────────────────────
        total_penalty = allergy_penalty + interaction_penalty + history_penalty + risk_penalty
        safety_score  = max(0, min(100, 100 - total_penalty))

        all_issues = allergy_issues + interaction_issues + history_issues

        return VerificationResult(
            safety_score       = safety_score,
            is_safe            = safety_score >= 50,
            allergy_issues     = allergy_issues,
            interaction_issues = interaction_issues,
            history_issues     = history_issues,
            risk_factors       = risk_factors,
            risk_label         = risk_label,
            all_issues         = all_issues,
            penalties          = {
                "allergy"    : allergy_penalty,
                "interaction": interaction_penalty,
                "history"    : history_penalty,
                "risk"       : risk_penalty,
                "total"      : total_penalty,
            },
        )
