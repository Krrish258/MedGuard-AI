"""
MedGuard-AI — Evidence Extractor
Parses a raw patient JSON dict into a structured, normalised
EvidenceBundle used throughout the agent reasoning loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Medication:
    name: str
    dose: str
    frequency: str

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Medication":
        return cls(
            name=d.get("name", "").strip(),
            dose=d.get("dose", "").strip(),
            frequency=d.get("frequency", "").strip(),
        )

    def __str__(self) -> str:
        return f"{self.name} {self.dose} ({self.frequency})"


@dataclass
class EvidenceBundle:
    """Structured patient evidence ready for agent reasoning."""
    patient_id: str
    age: int
    raw_symptoms: list[str]
    medical_history: list[str]
    current_medications: list[Medication]
    allergies: list[str]

    # Populated downstream by SymptomMatcher
    matched_symptoms: list[str] = field(default_factory=list)

    # Risk profile (computed)
    @property
    def age_risk(self) -> str:
        if self.age < 2:
            return "neonate"
        if self.age < 12:
            return "paediatric"
        if self.age < 18:
            return "adolescent"
        if self.age < 65:
            return "adult"
        return "elderly"

    @property
    def comorbidity_count(self) -> int:
        return len(self.medical_history)

    @property
    def medication_names(self) -> list[str]:
        return [m.name for m in self.current_medications]

    def summary(self) -> str:
        meds = ", ".join(str(m) for m in self.current_medications) or "None"
        return (
            f"Patient {self.patient_id} | Age: {self.age} ({self.age_risk})\n"
            f"Symptoms   : {', '.join(self.raw_symptoms)}\n"
            f"History    : {', '.join(self.medical_history) or 'None'}\n"
            f"Medications: {meds}\n"
            f"Allergies  : {', '.join(self.allergies) or 'None'}"
        )


class EvidenceExtractor:
    """
    Parses a raw patient JSON dict into an EvidenceBundle.
    Validates required fields and normalises string values.
    """

    REQUIRED_FIELDS = {"patient_id", "age", "symptoms", "medical_history",
                       "current_medications", "allergies"}

    def extract(self, patient_data: dict[str, Any]) -> EvidenceBundle:
        """
        Parse and validate patient JSON.

        Args:
            patient_data: dict conforming to patient_schema.json

        Returns:
            EvidenceBundle

        Raises:
            ValueError: if required fields are missing or invalid
        """
        self._validate(patient_data)

        return EvidenceBundle(
            patient_id=str(patient_data["patient_id"]).strip(),
            age=int(patient_data["age"]),
            raw_symptoms=[s.strip().lower() for s in patient_data["symptoms"] if s.strip()],
            medical_history=[h.strip().lower() for h in patient_data["medical_history"]],
            current_medications=[
                Medication.from_dict(m) for m in patient_data["current_medications"]
            ],
            allergies=[a.strip() for a in patient_data["allergies"]],
        )

    def _validate(self, data: dict[str, Any]) -> None:
        missing = self.REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"Patient data missing required fields: {missing}")
        if not isinstance(data["age"], int) or not (0 <= data["age"] <= 130):
            raise ValueError(f"Invalid age: {data['age']}")
        if not data["symptoms"]:
            raise ValueError("At least one symptom must be provided.")
