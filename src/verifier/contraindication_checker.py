"""
MedGuard-AI — Contraindication Checker
Checks for allergy conflicts and medical history contraindications
against a proposed treatment drug.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

ALLERGY_FILE     = ROOT / "knowledge_base" / "allergy_map.json"
CONTRAINDIC_FILE = ROOT / "knowledge_base" / "history_contraindications.json"

SEVERITY_PENALTY = {"CRITICAL": 45, "MAJOR": 30, "MODERATE": 12, "MINOR": 4}


class ContraindicationChecker:
    def __init__(self) -> None:
        with open(ALLERGY_FILE) as f:
            self._allergy_db: dict = json.load(f)
        with open(CONTRAINDIC_FILE) as f:
            self._history_db: dict = json.load(f)

    def _normalise(self, s: str) -> str:
        return s.strip().lower()

    def check_allergies(
        self,
        proposed_drug: str,
        patient_allergies: list[str],
    ) -> tuple[int, list[dict]]:
        """Check proposed drug against patient allergy list."""
        drug_lower = self._normalise(proposed_drug)
        issues: list[dict] = []
        penalty = 0

        for allergen in patient_allergies:
            allergen_key = None
            # Find matching allergen in DB (flexible lookup)
            for key in self._allergy_db:
                if key.startswith("_"):
                    continue
                if self._normalise(key) in self._normalise(allergen) or \
                   self._normalise(allergen) in self._normalise(key):
                    allergen_key = key
                    break

            if not allergen_key:
                continue

            allergy_data   = self._allergy_db[allergen_key]
            avoid_drugs    = [self._normalise(d) for d in allergy_data.get("avoid_drugs", [])]
            avoid_classes  = [self._normalise(c) for c in allergy_data.get("avoid_classes", [])]

            # Check if proposed drug matches avoid list
            drug_in_avoid = any(
                drug_lower in ad or ad in drug_lower
                for ad in avoid_drugs
            )
            class_match = any(
                cl in drug_lower or drug_lower in cl
                for cl in avoid_classes
            )

            if drug_in_avoid or class_match:
                sev = allergy_data.get("severity", "MAJOR")
                pen = SEVERITY_PENALTY.get(sev, 30)
                penalty += pen
                issues.append({
                    "type"         : "allergy_conflict",
                    "severity"     : sev,
                    "allergen"     : allergen,
                    "proposed_drug": proposed_drug,
                    "safe_alternatives": allergy_data.get("safe_alternatives", []),
                    "penalty"      : pen,
                    "message"      : (
                        f"Patient has reported allergy to '{allergen}'. "
                        f"'{proposed_drug}' belongs to an avoided drug class: "
                        f"{allergy_data.get('avoid_classes', [])}."
                    ),
                })

        return min(penalty, 60), issues

    def check_history(
        self,
        proposed_drug: str,
        medical_history: list[str],
    ) -> tuple[int, list[dict]]:
        """Check proposed drug against patient medical history contraindications."""
        drug_lower = self._normalise(proposed_drug)
        issues: list[dict] = []
        penalty = 0

        for condition in medical_history:
            cond_lower = self._normalise(condition)

            # Find matching condition in contraindication DB
            for cond_key, cond_data in self._history_db.items():
                if cond_key.startswith("_"):
                    continue
                if cond_lower not in self._normalise(cond_key) and \
                   self._normalise(cond_key) not in cond_lower:
                    continue

                contraindicated = [
                    self._normalise(d)
                    for d in cond_data.get("contraindicated_drugs", [])
                ]
                drug_matches = any(
                    drug_lower in cd or cd in drug_lower
                    for cd in contraindicated
                )

                if drug_matches:
                    sev = cond_data.get("severity", "MAJOR")
                    pen = SEVERITY_PENALTY.get(sev, 30)
                    penalty += pen
                    issues.append({
                        "type"         : "history_contraindication",
                        "severity"     : sev,
                        "condition"    : condition,
                        "proposed_drug": proposed_drug,
                        "reason"       : cond_data.get("reason", ""),
                        "safe_alternatives": cond_data.get("safe_alternatives", []),
                        "penalty"      : pen,
                        "message"      : (
                            f"'{proposed_drug}' is contraindicated in patients with "
                            f"'{condition}': {cond_data.get('reason', '')}."
                        ),
                    })

        return min(penalty, 60), issues
