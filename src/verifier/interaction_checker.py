"""
MedGuard-AI — Drug Interaction Checker
Checks for drug-drug interactions between a proposed treatment
and the patient's current medications.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INTERACTIONS_FILE = ROOT / "knowledge_base" / "drug_interactions.json"

SEVERITY_PENALTY = {"CRITICAL": 40, "MAJOR": 25, "MODERATE": 10, "MINOR": 3}


class InteractionChecker:
    def __init__(self) -> None:
        self._db: list[dict] = []
        self._load()

    def _load(self) -> None:
        with open(INTERACTIONS_FILE) as f:
            data = json.load(f)
        self._db = data.get("interactions", [])

    def check(
        self,
        proposed_drug: str,
        current_medications: list[str],
    ) -> tuple[int, list[dict]]:
        """
        Check for interactions between proposed_drug and all current meds.

        Returns:
            penalty : total score penalty (0–100)
            issues  : list of interaction issue dicts
        """
        proposed_lower = proposed_drug.lower()
        issues: list[dict] = []
        penalty = 0

        for med in current_medications:
            med_lower = med.lower()
            for entry in self._db:
                da = entry["drug_a"].lower()
                db = entry["drug_b"].lower()

                if (proposed_lower in da or da in proposed_lower) and \
                   (med_lower in db or db in med_lower):
                    match = True
                elif (proposed_lower in db or db in proposed_lower) and \
                     (med_lower in da or da in med_lower):
                    match = True
                else:
                    match = False

                if match:
                    sev  = entry.get("severity", "MINOR")
                    pen  = SEVERITY_PENALTY.get(sev, 3)
                    penalty += pen
                    issues.append({
                        "type"        : "drug_drug_interaction",
                        "severity"    : sev,
                        "drug_a"      : entry["drug_a"],
                        "drug_b"      : entry["drug_b"],
                        "effect"      : entry.get("effect", ""),
                        "management"  : entry.get("management", ""),
                        "penalty"     : pen,
                    })

        return min(penalty, 60), issues  # cap at 60 pts
