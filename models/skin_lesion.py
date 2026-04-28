"""
models/skin_lesion.py
---------------------
SkinLesion domain object.

Holds all state for a single lesion scan: the raw image, each processing
artefact (enhanced, mask), extracted features, and the final risk verdict.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np


class SkinLesion:
    """
    Represents one skin lesion image and all derived processing results.

    Attributes
    ----------
    image_path : str
        Absolute path of the source image file.
    body_location : str
        Anatomical region, e.g. "Back", "Arm".
    patient_id : str | None
        Set by Patient.add_lesion().
    raw_image : np.ndarray | None
        BGR image as loaded from disk (shape H×W×3, dtype uint8).
    enhanced_image : np.ndarray | None
        Output of the Enhancer pipeline.
    mask : np.ndarray | None
        Binary mask (uint8, 0/255) isolating the lesion ROI.
    features : dict | None
        ABCDE feature dict produced by FeatureExtractor.
    risk_score : float | None
        Classifier probability in [0, 1].
    risk_level : str | None
        "Low" | "Medium" | "High".
    report : ScanReport | None
        Back-reference to the generated report (set by SkinScanApp).
    """

    def __init__(
        self,
        image_path: str,
        body_location: str = "Unknown",
    ) -> None:
        self.image_path: str = image_path
        self.body_location: str = body_location
        self.patient_id: Optional[str] = None

        # Processing artefacts – populated progressively
        self.raw_image: Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.features: Optional[dict] = None
        self.risk_score: Optional[float] = None
        self.risk_level: Optional[str] = None
        self.report = None  # ScanReport (avoid circular import)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def image_name(self) -> str:
        """Filename without directory path."""
        return os.path.basename(self.image_path)

    @property
    def working_image(self) -> np.ndarray:
        """
        Return the best available intermediate image for processing.
        Falls back through enhanced → raw (never None if raw_image is set).
        """
        if self.enhanced_image is not None:
            return self.enhanced_image
        if self.raw_image is not None:
            return self.raw_image
        raise ValueError("No image loaded – call load_image first.")

    @property
    def is_analysed(self) -> bool:
        return self.risk_level is not None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SkinLesion(name={self.image_name!r}, "
            f"risk={self.risk_level!r}, score={self.risk_score})"
        )
