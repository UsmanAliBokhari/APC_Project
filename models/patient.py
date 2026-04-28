"""
models/patient.py
-----------------
Patient domain object.

A Patient is the top-level aggregate: it owns zero or more SkinLesion
objects and can compute an overall risk level across all scanned lesions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.skin_lesion import SkinLesion


@dataclass
class Patient:
    """
    Represents a clinic patient.

    Attributes
    ----------
    patient_id : str
        Unique identifier, e.g. "PT-0001".
    name : str
        Full name.
    age : int
        Age in years.
    gender : str
        "Male" | "Female" | "Other" | "Unspecified".
    body_location : str
        Anatomical region of the lesion, e.g. "Back", "Arm".
    lesions : list[SkinLesion]
        Lesions added during the session (populated via add_lesion).
    """

    patient_id: str
    name: str
    age: int
    gender: str = "Unspecified"
    body_location: str = "Unknown"
    lesions: list = field(default_factory=list)  # list[SkinLesion]

    # ------------------------------------------------------------------
    # Lesion management
    # ------------------------------------------------------------------

    def add_lesion(self, lesion: "SkinLesion") -> None:
        """Append a lesion and back-link it to this patient."""
        lesion.patient_id = self.patient_id
        self.lesions.append(lesion)

    # ------------------------------------------------------------------
    # Risk aggregation
    # ------------------------------------------------------------------

    def overall_risk_level(self) -> str:
        """
        Return the highest risk level across all analysed lesions.

        Priority: High > Medium > Low > Pending
        """
        levels = [l.risk_level for l in self.lesions if l.risk_level]
        if "High" in levels:
            return "High"
        if "Medium" in levels:
            return "Medium"
        if "Low" in levels:
            return "Low"
        return "Pending"

    def analysed_lesions(self) -> list:
        """Return only lesions that have been fully analysed."""
        return [l for l in self.lesions if l.risk_level is not None]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Patient(id={self.patient_id!r}, name={self.name!r}, "
            f"age={self.age}, lesions={len(self.lesions)})"
        )
