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

    Parameters
    ----------
    image_path    : str  — absolute path of the source image file
    body_location : str  — anatomical region e.g. "Back", "Arm"

    Attributes
    ----------
    raw_image      : BGR image as loaded from disk (H×W×3, uint8)
    enhanced_image : output of the Enhancer pipeline
    mask           : binary mask (uint8, 0/255) isolating the lesion ROI
    features       : ABCDE feature dict produced by FeatureExtractor
    risk_score     : classifier probability in [0, 1]
    risk_level     : "Low" | "Medium" | "High"
    """

    def __init__(self, image_path: str, body_location: str = "Unknown") -> None:
        self.image_path    = image_path
        self.body_location = body_location

        # processing artefacts — populated progressively as the pipeline runs
        self.raw_image:      Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        self.mask:           Optional[np.ndarray] = None
        self.features:       Optional[dict]        = None
        self.risk_score:     Optional[float]       = None
        self.risk_level:     Optional[str]         = None
        self.report                                = None 

    @property
    def image_name(self) -> str:
        """Filename without directory path."""
        return os.path.basename(self.image_path)

    @property
    def working_image(self) -> np.ndarray:
        """Best available image for processing — falls back enhanced → raw."""
        if self.enhanced_image is not None:
            return self.enhanced_image
        if self.raw_image is not None:
            return self.raw_image
        raise ValueError("No image loaded — call load_image first.")

    @property
    def is_analysed(self) -> bool:
        """True once a risk level has been assigned."""
        return self.risk_level is not None

    def __repr__(self) -> str:
        return (f"SkinLesion(name={self.image_name!r}, "
                f"risk={self.risk_level!r}, score={self.risk_score})")