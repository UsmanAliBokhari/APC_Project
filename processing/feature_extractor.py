"""
processing/feature_extractor.py
--------------------------------
FeatureExtractor(ImageProcessor) – compute ABCDE-rule features.

ABCDE Rule
----------
A – Asymmetry       : how non-symmetric the lesion contour is
B – Border          : irregularity / jaggedness of the boundary
C – Colour          : variance of colour channels within the lesion
D – Diameter        : relative size (normalised to image diagonal)
E – Evolving (proxy): GLCM texture contrast (stands in for change over time)

All returned values are floats.  The feature dict is compatible with the
scikit-learn pipeline in detection/detector.py.
"""

from __future__ import annotations
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from processing.image_processor import ImageProcessor


class FeatureExtractor(ImageProcessor):
    """
    Extract ABCDE features from an enhanced lesion image + binary mask.

    Parameters
    ----------
    image : np.ndarray
        Enhanced BGR image (uint8).
    mask : np.ndarray
        Binary mask (uint8, 0/255) from Segmenter.  If None, the whole
        image is treated as the lesion.
    """

    def __init__(self, image: np.ndarray, mask: np.ndarray | None = None) -> None:
        self._mask_input = mask
        super().__init__(image)

    # ------------------------------------------------------------------
    # ImageProcessor interface
    # ------------------------------------------------------------------

    def _process(self, **_) -> None:
        """
        Compute all features and store in self._result (a dict).
        self._result is typed as np.ndarray in the base class, but the
        property is overridden below to return a dict.
        """
        mask = self._mask_input
        if mask is None:
            mask = np.ones(self._source.shape[:2], dtype=np.uint8) * 255

        features: dict = {}
        features.update(self._asymmetry(mask))
        features.update(self._border(mask))
        features.update(self._colour(self._source, mask))
        features.update(self._diameter(mask, self._source.shape))
        features.update(self._texture(self._source, mask))

        # Store as attribute; override result property to return dict
        self._feature_dict = features
        self._result = np.array(list(features.values()), dtype=np.float32)

    # ------------------------------------------------------------------
    # result override – return dict instead of raw array
    # ------------------------------------------------------------------

    @property
    def result(self) -> dict:  # type: ignore[override]
        return self._feature_dict

    # ------------------------------------------------------------------
    # A – Asymmetry
    # ------------------------------------------------------------------

    @staticmethod
    def _asymmetry(mask: np.ndarray) -> dict:
        """
        Flip the mask along each axis and measure overlap.
        Score 0 → perfect symmetry; 1 → maximally asymmetric.
        """
        h, w = mask.shape

        # Horizontal flip
        flipped_h = cv2.flip(mask, 1)
        diff_h = cv2.bitwise_xor(mask, flipped_h)
        score_h = diff_h.sum() / (mask.sum() + 1e-6)

        # Vertical flip
        flipped_v = cv2.flip(mask, 0)
        diff_v = cv2.bitwise_xor(mask, flipped_v)
        score_v = diff_v.sum() / (mask.sum() + 1e-6)

        asymmetry = float(np.clip((score_h + score_v) / 2.0, 0.0, 1.0))
        return {"asymmetry": asymmetry}

    # ------------------------------------------------------------------
    # B – Border Irregularity
    # ------------------------------------------------------------------

    @staticmethod
    def _border(mask: np.ndarray) -> dict:
        """
        Compactness index: 4π·Area / Perimeter².
        1.0 = perfect circle; lower → more irregular.
        We invert so higher = more irregular.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return {"border_irregularity": 0.0}

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, closed=True))

        if perimeter < 1e-6:
            return {"border_irregularity": 0.0}

        compactness = (4.0 * np.pi * area) / (perimeter ** 2)
        irregularity = float(np.clip(1.0 - compactness, 0.0, 1.0))
        return {"border_irregularity": irregularity}

    # ------------------------------------------------------------------
    # C – Colour
    # ------------------------------------------------------------------

    @staticmethod
    def _colour(img: np.ndarray, mask: np.ndarray) -> dict:
        """
        Compute mean and std of each BGR channel within the masked region.
        High std in any channel signals multicolour (higher risk).
        """
        features: dict = {}
        channel_names = ["b", "g", "r"]

        for idx, ch_name in enumerate(channel_names):
            channel = img[:, :, idx]
            roi_pixels = channel[mask > 0].astype(np.float32)

            if roi_pixels.size == 0:
                features[f"color_mean_{ch_name}"] = 0.0
                features[f"color_std_{ch_name}"]  = 0.0
            else:
                features[f"color_mean_{ch_name}"] = float(roi_pixels.mean())
                features[f"color_std_{ch_name}"]  = float(roi_pixels.std())

        return features

    # ------------------------------------------------------------------
    # D – Diameter (relative size)
    # ------------------------------------------------------------------

    @staticmethod
    def _diameter(mask: np.ndarray, img_shape: tuple) -> dict:
        """
        Estimate the effective diameter of the lesion as a fraction of
        the image diagonal.  Threshold at 6mm equivalent is ~0.05 on
        typical dermoscopy images.
        """
        h, w = img_shape[:2]
        diagonal = float(np.sqrt(h ** 2 + w ** 2))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {"diameter_norm": 0.0}

        cnt = max(contours, key=cv2.contourArea)
        _, (box_w, box_h), _ = cv2.minAreaRect(cnt)
        effective_diam = float(max(box_w, box_h))

        diameter_norm = float(np.clip(effective_diam / diagonal, 0.0, 1.0))
        return {"diameter_norm": diameter_norm}

    # ------------------------------------------------------------------
    # E – Texture (GLCM contrast as proxy for Evolving)
    # ------------------------------------------------------------------

    @staticmethod
    def _texture(img: np.ndarray, mask: np.ndarray) -> dict:
        """
        GLCM (Gray-Level Co-occurrence Matrix) texture features within
        the lesion region.  We use contrast and homogeneity.

        scikit-image's graycomatrix operates on integer arrays; we
        quantise to 64 levels for speed.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Zero-out background pixels, quantise to 64 levels
        roi = np.where(mask > 0, gray, 0).astype(np.uint8)
        roi_q = (roi // 4).astype(np.uint8)   # 0-63

        try:
            glcm = graycomatrix(
                roi_q,
                distances=[1, 2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=64,
                symmetric=True,
                normed=True,
            )
            contrast    = float(graycoprops(glcm, "contrast").mean())
            homogeneity = float(graycoprops(glcm, "homogeneity").mean())
            energy      = float(graycoprops(glcm, "energy").mean())
            correlation = float(graycoprops(glcm, "correlation").mean())
        except Exception:
            contrast = homogeneity = energy = correlation = 0.0

        # Normalise contrast to ~[0, 1] range for a dermoscopy image
        return {
            "texture_contrast":    float(np.clip(contrast / 200.0, 0.0, 2.0)),
            "texture_homogeneity": float(np.clip(homogeneity, 0.0, 1.0)),
            "texture_energy":      float(np.clip(energy, 0.0, 1.0)),
            "texture_correlation": float(np.clip(abs(correlation), 0.0, 1.0)),
        }
