"""
MedGuard-AI — Decision Module (Layer 3)
Maps the safety score and verification result to a final decision.

Decision thresholds:
  APPROVED  → score ≥ 85, no CRITICAL issues, confidence ≥ 0.60
  WARNING   → score 70–84, or non-critical issues found
  ABSTAIN   → score < 70, CRITICAL issue, or confidence < 0.60
"""

from __future__ import annotations
from dataclasses import dataclass

from .verifier.causal_verifier import VerificationResult
from .agent.reasoning_agent import AgentResult

APPROVED_THRESHOLD  = 85
WARNING_THRESHOLD   = 70
CONFIDENCE_MINIMUM  = 0.60


@dataclass
class FinalDecision:
    decision   : str    # 'approved' | 'warning' | 'abstain'
    message    : str
    safety_score: int
    confidence : float
    color_code : str    # for CLI display


class DecisionModule:
    """Translates verification output into a final clinical decision."""

    def decide(
        self,
        agent_result: AgentResult,
        verification: VerificationResult,
    ) -> FinalDecision:
        score      = verification.safety_score
        confidence = agent_result.confidence
        has_critical = verification.has_critical
        low_conf     = agent_result.low_confidence

        # ── ABSTAIN conditions ────────────────────────────────────────────────
        if has_critical:
            return FinalDecision(
                decision    = "abstain",
                message     = (
                    "⛔ CRITICAL safety issue detected. Treatment is CONTRAINDICATED. "
                    "Immediate clinical review required."
                ),
                safety_score=score,
                confidence  =confidence,
                color_code  ="red",
            )

        if score < WARNING_THRESHOLD:
            return FinalDecision(
                decision    = "abstain",
                message     = (
                    f"🔴 Safety score too low ({score}/100). "
                    "Treatment plan carries unacceptable risk. "
                    "Alternative therapy strongly recommended."
                ),
                safety_score=score,
                confidence  =confidence,
                color_code  ="red",
            )

        if low_conf:
            return FinalDecision(
                decision    = "abstain",
                message     = (
                    f"🟡 Diagnostic confidence insufficient ({confidence:.1%}). "
                    "The system cannot make a confident diagnosis from the provided symptoms. "
                    "Further clinical investigation required."
                ),
                safety_score=score,
                confidence  =confidence,
                color_code  ="yellow",
            )

        # ── WARNING conditions ────────────────────────────────────────────────
        if score < APPROVED_THRESHOLD or verification.issue_count > 0:
            n_issues = verification.issue_count
            issue_summary = (
                f"{n_issues} safety concern(s) noted — "
                + ", ".join(
                    f"{i.get('type','issue')} [{i.get('severity','?')}]"
                    for i in verification.all_issues[:3]
                )
            ) if n_issues else f"Safety score {score}/100 below optimal threshold."

            return FinalDecision(
                decision    = "warning",
                message     = (
                    f"⚠️  Treatment plan accepted with caution. {issue_summary}. "
                    "Clinical review recommended before administration."
                ),
                safety_score=score,
                confidence  =confidence,
                color_code  ="yellow",
            )

        # ── APPROVED ──────────────────────────────────────────────────────────
        return FinalDecision(
            decision    = "approved",
            message     = (
                f"✅ Treatment plan verified safe (score {score}/100, "
                f"confidence {confidence:.1%}). "
                "No significant safety concerns detected."
            ),
            safety_score=score,
            confidence  =confidence,
            color_code  ="green",
        )
