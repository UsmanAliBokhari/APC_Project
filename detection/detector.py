"""
detection/detector.py
---------------------
Detector – loads the trained RandomForestClassifier and classifies lesions.

The model is expected at models/detector.pkl (created by utils/model_setup.py
on first run).  The Detector exposes a single public method:

    risk_score, risk_level = detector.predict(feature_dict)

Risk thresholds
---------------
    score < 0.35  → "Low"
    0.35 ≤ score < 0.65 → "Medium"
    score ≥ 0.65  → "High"
"""

from __future__ import annotations
import os

import joblib
import numpy as np

# Path to serialised model (relative to project root)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "detector.pkl")

# ABCDE feature order must match the order used during training
# (see utils/model_setup.py → _feature_vector)
_FEATURE_ORDER = [
    "asymmetry",
    "border_irregularity",
    "color_mean_b",
    "color_mean_g",
    "color_mean_r",
    "color_std_b",
    "color_std_g",
    "color_std_r",
    "diameter_norm",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
]


class Detector:
    """
    Wraps a trained scikit-learn pipeline for malignancy risk scoring.

    The model is loaded lazily on the first call to `predict`.
    """

    def __init__(self) -> None:
        self._pipeline = None   # loaded on first use

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> tuple[float, str]:
        """
        Classify one lesion.

        Parameters
        ----------
        features : dict
            Feature dict from FeatureExtractor.result.

        Returns
        -------
        (risk_score, risk_level)
            risk_score : float in [0, 1]  (probability of malignancy)
            risk_level : "Low" | "Medium" | "High"
        """
        if self._pipeline is None:
            self._load_model()

        vec = self._dict_to_vector(features)
        prob = float(self._pipeline.predict_proba(vec)[0][1])
        level = self._score_to_level(prob)
        return prob, level

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path = os.path.abspath(_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Run Main.py to auto-generate the model, or call "
                "utils.model_setup.ensure_model_exists() first."
            )
        self._pipeline = joblib.load(model_path)

    @staticmethod
    def _dict_to_vector(features: dict) -> np.ndarray:
        """
        Convert feature dict → (1, N) float32 array in the correct order.
        Missing keys default to 0.0.
        """
        vec = np.array(
            [features.get(k, 0.0) for k in _FEATURE_ORDER],
            dtype=np.float32,
        ).reshape(1, -1)
        return vec

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 0.65:
            return "High"
        if score >= 0.35:
            return "Medium"
        return "Low"
