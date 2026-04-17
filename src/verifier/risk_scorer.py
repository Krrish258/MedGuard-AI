"""
MedGuard-AI — Outcome Risk Scorer
Computes a patient-level risk penalty based on age profile
and comorbidity burden. This adjusts the safety score to reflect
patient-specific risk independent of the proposed treatment.
"""

from __future__ import annotations


# Comorbidity conditions that compound overall risk
HIGH_RISK_CONDITIONS = {
    "hypertension", "type 2 diabetes", "diabetes",
    "heart failure", "renal failure", "liver failure", "cirrhosis",
    "hiv", "aids", "cancer", "copd", "asthma",
    "immunocompromised", "chronic kidney disease",
}


class RiskScorer:
    """
    Computes an outcome risk penalty (0–22) based on:
      - Patient age (neonate/infant/elderly add risk)
      - Number and severity of comorbidities
    """

    def score(
        self,
        age: int,
        medical_history: list[str],
        confidence: float,
    ) -> tuple[int, str, list[str]]:
        """
        Compute risk penalty and risk label.

        Args:
            age            : patient age in years
            medical_history: list of patient conditions
            confidence     : model confidence in diagnosis (0.0–1.0)

        Returns:
            penalty        : int 0–22
            risk_label     : 'LOW', 'MODERATE', 'HIGH'
            risk_factors   : list of identified risk factors
        """
        penalty = 0
        risk_factors: list[str] = []

        # ── Age risk ──────────────────────────────────────────────────────────
        if age == 0:  # Proxy for neonate (≤28 days) / <1 year
            penalty += 10
            risk_factors.append(f"Neonate/Infant (age {age}) — extreme pharmacokinetic variability (WHO/AAP)")
        elif age < 2:
            penalty += 7
            risk_factors.append(f"Toddler (age {age}) — immature renal/hepatic metabolism")
        elif age < 12:
            penalty += 4
            risk_factors.append(f"Paediatric patient (age {age}) — weight-based dosing needed")
        elif age >= 80:
            penalty += 7
            risk_factors.append(f"Very elderly ({age}) — high polypharmacy & falls risk (Beers Criteria)")
        elif age >= 65:
            penalty += 4
            risk_factors.append(f"Elderly patient (age {age}) — monitor closely (Beers Criteria)")

        # ── Comorbidity burden ──────────────────────────────────────────────
        high_risk_comorbidities = [
            c for c in medical_history
            if any(hrc in c.lower() for hrc in HIGH_RISK_CONDITIONS)
        ]
        comorbidity_count = len(high_risk_comorbidities)

        if comorbidity_count >= 3:
            penalty += 8
            risk_factors.append(f"Multiple high-risk comorbidities ({comorbidity_count}) (CCI equivalent)")
        elif comorbidity_count == 2:
            penalty += 5
            risk_factors.append(f"Two high-risk comorbidities: {high_risk_comorbidities}")
        elif comorbidity_count == 1:
            penalty += 3
            risk_factors.append(f"Comorbidity: {high_risk_comorbidities[0]}")

        # ── Low model confidence adds outcome uncertainty ─────────────────────
        if confidence < 0.60:
            penalty += 4
            risk_factors.append(
                f"Low diagnostic confidence ({confidence:.1%}) — diagnosis uncertain"
            )
        elif confidence < 0.75:
            penalty += 2
            risk_factors.append(
                f"Moderate diagnostic confidence ({confidence:.1%})"
            )

        penalty = min(penalty, 22)

        if penalty >= 14:
            label = "HIGH"
        elif penalty >= 7:
            label = "MODERATE"
        else:
            label = "LOW"

        return penalty, label, risk_factors
