"""
models/scan_report.py
---------------------
ScanReport domain object.

Encapsulates a completed analysis: patient metadata, lesion features,
risk verdict, and auto-generated recommendations.  Can export to plain
text for archival.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.patient import Patient
    from models.skin_lesion import SkinLesion


# Recommendation rules keyed by risk level
_RECOMMENDATIONS: dict[str, list[str]] = {
    "Low": [
        "No immediate action required.",
        "Continue regular self-examination every 1–3 months.",
        "Use SPF 30+ sunscreen daily and avoid peak UV hours.",
        "Book a routine dermatology review within 12 months.",
    ],
    "Medium": [
        "Schedule a dermatology appointment within 4–6 weeks.",
        "Photograph the lesion monthly to track changes.",
        "Watch for changes in size, shape, colour, or bleeding.",
        "Avoid direct sun exposure on the affected area.",
        "Do not attempt self-treatment.",
    ],
    "High": [
        "⚠️  Seek urgent dermatological evaluation (within 1 week).",
        "A biopsy may be required for definitive diagnosis.",
        "Avoid irritating or scratching the lesion.",
        "Inform your GP immediately and request a specialist referral.",
        "This result is a screening aid – always confirm with a clinician.",
    ],
}


class ScanReport:
    """
    A completed analysis report for one SkinLesion.

    Parameters
    ----------
    patient : Patient
    lesion  : SkinLesion
    risk_score : float   – classifier output probability [0, 1]
    risk_level : str     – "Low" | "Medium" | "High"
    features   : dict    – raw ABCDE feature dict from FeatureExtractor
    """

    def __init__(
        self,
        patient: "Patient",
        lesion: "SkinLesion",
        risk_score: float,
        risk_level: str,
        features: dict,
    ) -> None:
        self.patient = patient
        self.lesion = lesion
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.features = features
        self.timestamp: datetime = datetime.now()
        self.recommendations: list[str] = _RECOMMENDATIONS.get(risk_level, [])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_txt(self, path: str) -> str:
        """
        Write a human-readable report to *path* and return the written path.
        """
        lines: list[str] = [
            "=" * 60,
            "  SkinScan  –  Melanoma Screening Report",
            "=" * 60,
            f"Generated : {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "── Patient ─────────────────────────────────────────────",
            f"  ID       : {self.patient.patient_id}",
            f"  Name     : {self.patient.name}",
            f"  Age      : {self.patient.age}",
            f"  Gender   : {self.patient.gender}",
            f"  Location : {self.patient.body_location}",
            "",
            "── Image ───────────────────────────────────────────────",
            f"  File     : {self.lesion.image_name}",
            f"  Path     : {self.lesion.image_path}",
            "",
            "── Risk Assessment ─────────────────────────────────────",
            f"  Level    : {self.risk_level}",
            f"  Score    : {self.risk_score:.4f}",
            "",
            "── ABCDE Features ──────────────────────────────────────",
        ]

        # Pretty-print every feature
        for k, v in sorted(self.features.items()):
            lines.append(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

        lines += [
            "",
            "── Recommendations ─────────────────────────────────────",
        ]
        lines += [f"  • {rec}" for rec in self.recommendations]
        lines += [
            "",
            "─" * 60,
            "DISCLAIMER: This tool is a decision-support aid only.",
            "It does NOT replace clinical diagnosis by a qualified",
            "dermatologist.  Always consult a medical professional.",
            "=" * 60,
        ]

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        return path

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ScanReport(patient={self.patient.name!r}, "
            f"risk={self.risk_level!r}, score={self.risk_score:.3f})"
        )
