"""
processing/feature_extractor.py
--------------------------------
FeatureExtractor(ImageProcessor) – compute ABCDE-rule features.

A – Asymmetry  : how non-symmetric the lesion contour is
B – Border     : irregularity / jaggedness of the boundary
C – Colour     : variance of colour channels within the lesion
D – Diameter   : relative size normalised to image diagonal
E – Evolving   : GLCM texture contrast (stands in for change over time)
"""

from __future__ import annotations
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from processing.image_processor import ImageProcessor


class FeatureExtractor(ImageProcessor):
    """
    Extract ABCDE-rule features from an enhanced lesion image and binary mask.

    Parameters
    ----------
    image : np.ndarray
        Enhanced RGB image (uint8, HxWx3).
    mask : np.ndarray | None
        Binary mask (uint8, 0/255) from Segmenter.
        If None, the entire image is treated as the lesion.
    """

    def __init__(self, image: np.ndarray, mask: np.ndarray | None = None) -> None:
        self._mask_input = mask
        super().__init__(image)

    def _process(self, **_) -> None:
        """Compute all features and store in self._feature_dict."""
        mask = self._mask_input if self._mask_input is not None \
               else np.ones(self._source.shape[:2], dtype=np.uint8) * 255

        self._feature_dict = {
            **self._asymmetry(mask),
            **self._border(mask),
            **self._colour(self._source, mask),
            **self._diameter(mask, self._source.shape),
            **self._texture(self._source, mask),
        }
        self._result = np.array(list(self._feature_dict.values()), dtype=np.float32)

    @property
    def result(self) -> dict:
        """Return features as a dict instead of raw array."""
        return self._feature_dict
    # ------------------------------------------------------------------
    # A – Asymmetry
    # ------------------------------------------------------------------

    @staticmethod
    def _asymmetry(mask: np.ndarray) -> dict:
        """Flip mask on each axis and measure overlap — 0 = symmetric, 1 = asymmetric."""
        # bitwise_xor finds pixels that differ between original and flipped
        score_h = cv2.bitwise_xor(mask, cv2.flip(mask, 1)).sum() / (mask.sum() + 1e-6)
        score_v = cv2.bitwise_xor(mask, cv2.flip(mask, 0)).sum() / (mask.sum() + 1e-6)
        return {"asymmetry": float(np.clip((score_h + score_v) / 2.0, 0.0, 1.0))}

    # ------------------------------------------------------------------
    # B – Border Irregularity
    # ------------------------------------------------------------------

    @staticmethod
    def _border(mask: np.ndarray) -> dict:
        """Compactness index — 1 = circle, lower = more irregular."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"border_irregularity": 0.0}

        cnt       = max(contours, key=cv2.contourArea)
        area      = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, closed=True))

        if perimeter < 1e-6:
            return {"border_irregularity": 0.0}

        irregularity = 1.0 - (4.0 * np.pi * area) / (perimeter ** 2)
        return {"border_irregularity": float(np.clip(irregularity, 0.0, 1.0))}
    # ------------------------------------------------------------------
    # C – Colour
    # ------------------------------------------------------------------

    @staticmethod
    def _colour(img: np.ndarray, mask: np.ndarray) -> dict:
        """Mean and std of each channel inside the mask — high std signals multicolour."""
        # list comprehension over channels 
        stats = {
            f"color_{stat}_{name}": float(fn(img[:, :, i][mask > 0].astype(np.float32)))
            if img[:, :, i][mask > 0].size > 0 else 0.0
            for i, name in enumerate(["b", "g", "r"])
            for stat, fn in [("mean", np.mean), ("std", np.std)]
        }
        return stats
 # ------------------------------------------------------------------
    # D – Diameter (relative size)
    # ------------------------------------------------------------------

    @staticmethod
    def _diameter(mask: np.ndarray, img_shape: tuple) -> dict:
        """Lesion diameter as a fraction of the image diagonal."""
        diagonal  = float(np.sqrt(img_shape[0]**2 + img_shape[1]**2))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"diameter_norm": 0.0}

        _, (box_w, box_h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        return {"diameter_norm": float(np.clip(max(box_w, box_h) / diagonal, 0.0, 1.0))}
# ------------------------------------------------------------------
    # E – Texture (GLCM contrast as proxy for Evolving)
    # ------------------------------------------------------------------

    @staticmethod
    def _texture(img: np.ndarray, mask: np.ndarray) -> dict:
        """GLCM texture contrast within the lesion region."""
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_q = (np.where(mask > 0, gray, 0) // 4).astype(np.uint8) # quantise to 64 levels

        try:
            glcm = graycomatrix(roi_q, distances=[1, 2],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=64, symmetric=True, normed=True)
            # map over properties 
            props = {p: float(graycoprops(glcm, p).mean())
                     for p in ["contrast", "homogeneity", "energy", "correlation"]}
        except Exception:
            props = {p: 0.0 for p in ["contrast", "homogeneity", "energy", "correlation"]}

        return {
            "texture_contrast":    float(np.clip(props["contrast"]          / 200.0, 0.0, 2.0)),
            "texture_homogeneity": float(np.clip(props["homogeneity"],        0.0, 1.0)),
            "texture_energy":      float(np.clip(props["energy"],             0.0, 1.0)),
            "texture_correlation": float(np.clip(abs(props["correlation"]),   0.0, 1.0)),
        }