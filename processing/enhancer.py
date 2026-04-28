"""
processing/enhancer.py
-----------------------
Enhancer(ImageProcessor) – image quality improvement pipeline.

Supported methods
-----------------
"clahe"     : Contrast-Limited Adaptive Histogram Equalisation only.
"denoise"   : Bilateral / Non-local means denoising only.
"sharpen"   : Unsharp-mask sharpening only.
"pipeline"  : CLAHE → denoise → sharpen  (default, best for dermoscopy).
"""

from __future__ import annotations
import cv2
import numpy as np

from processing.image_processor import ImageProcessor


class Enhancer(ImageProcessor):
    """
    Improve image contrast, reduce noise, and sharpen edges.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image.
    method : str
        One of "clahe" | "denoise" | "sharpen" | "pipeline".
        Defaults to "pipeline".
    clip_limit : float
        CLAHE clip limit (higher → more contrast). Default 2.0.
    tile_size : int
        CLAHE grid tile size (default 8 → 8×8 grid).
    """

    def __init__(
        self,
        image: np.ndarray,
        method: str = "pipeline",
        clip_limit: float = 2.0,
        tile_size: int = 8,
    ) -> None:
        self._method = method
        self._clip_limit = clip_limit
        self._tile_size = tile_size
        super().__init__(image)

    # ------------------------------------------------------------------
    # ImageProcessor interface
    # ------------------------------------------------------------------

    def _process(self, **_) -> None:
        dispatch = {
            "clahe":    self._apply_clahe,
            "denoise":  self._apply_denoise,
            "sharpen":  self._apply_sharpen,
            "pipeline": self._apply_pipeline,
        }
        fn = dispatch.get(self._method)
        if fn is None:
            raise ValueError(
                f"Unknown method {self._method!r}. "
                f"Choose from: {list(dispatch)}"
            )
        self._result = fn(self._source)

    # ------------------------------------------------------------------
    # Individual enhancement steps
    # ------------------------------------------------------------------

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE in the LAB colour space: enhances luminance without
        shifting hue – important for preserving lesion colour cues.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_size, self._tile_size),
        )
        l_eq = clahe.apply(l_ch)

        merged = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_denoise(img: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising – reduces skin hair/speckle noise
        while preserving lesion boundary sharpness.
        """
        return cv2.fastNlMeansDenoisingColored(
            img, None,
            h=7,           # luminance filter strength
            hColor=7,      # colour component filter strength
            templateWindowSize=7,
            searchWindowSize=21,
        )

    @staticmethod
    def _apply_sharpen(img: np.ndarray) -> np.ndarray:
        """
        Unsharp-mask sharpening: subtracts a Gaussian-blurred version
        to accentuate high-frequency edge details.
        """
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        # addWeighted(src1, α, src2, β, γ):  result = α*img + β*(-blurred) + γ
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def _apply_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Full three-step pipeline: CLAHE → denoise → sharpen."""
        img = self._apply_clahe(img)
        img = self._apply_denoise(img)
        img = self._apply_sharpen(img)
        return img
