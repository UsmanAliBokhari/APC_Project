"""
processing/segmenter.py
------------------------
Segmenter(ImageProcessor) – isolate the lesion ROI from surrounding skin.

Supported methods
-----------------
"otsu"      : Global Otsu threshold on grayscale (fast, robust default).
"kmeans"    : K-means colour clustering, then select dominant lesion cluster.
"adaptive"  : Adaptive (block-wise) thresholding for uneven illumination.
"""

from __future__ import annotations
import cv2
import numpy as np

from processing.image_processor import ImageProcessor


class Segmenter(ImageProcessor):
    """
    Produce a binary mask (uint8, 0/255) of the lesion region.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image (typically the enhanced output).
    method : str
        "otsu" | "kmeans" | "adaptive".  Default: "otsu".
    k : int
        Number of clusters for K-means (ignored for other methods).
    """

    def __init__(
        self,
        image: np.ndarray,
        method: str = "otsu",
        k: int = 3,
    ) -> None:
        self._method = method
        self._k = k
        super().__init__(image)

    # ------------------------------------------------------------------
    # ImageProcessor interface
    # ------------------------------------------------------------------

    def _process(self, **_) -> None:
        dispatch = {
            "otsu":     self._segment_otsu,
            "kmeans":   self._segment_kmeans,
            "adaptive": self._segment_adaptive,
        }
        fn = dispatch.get(self._method)
        if fn is None:
            raise ValueError(
                f"Unknown method {self._method!r}. "
                f"Choose from: {list(dispatch)}"
            )
        mask = fn(self._source)
        self._result = self._postprocess(mask)

    # ------------------------------------------------------------------
    # Segmentation strategies
    # ------------------------------------------------------------------

    def _segment_otsu(self, img: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale, invert (lesions are typically darker),
        then apply Otsu's threshold.
        """
        gray = self.to_gray(img)
        # Gaussian blur reduces noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        return mask

    def _segment_kmeans(self, img: np.ndarray) -> np.ndarray:
        """
        Cluster pixels by colour; select the cluster(s) with lowest
        mean intensity (darkest = lesion).
        """
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20, 1.0,
        )
        _, labels, centers = cv2.kmeans(
            pixels, self._k, None, criteria, 5,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        # Rank clusters by mean luminance (darker = lesion)
        gray_centers = np.dot(centers, [0.114, 0.587, 0.299])
        lesion_label = int(np.argmin(gray_centers))

        mask_flat = np.where(labels.flatten() == lesion_label, 255, 0).astype(np.uint8)
        return mask_flat.reshape(img.shape[:2])

    def _segment_adaptive(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive (Gaussian) thresholding – better for uneven lighting.
        """
        gray = self.to_gray(img)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=4,
        )
        return mask

    # ------------------------------------------------------------------
    # Post-processing (morphological cleanup)
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(mask: np.ndarray) -> np.ndarray:
        """
        Remove small artefacts and fill holes using morphological ops.

        Steps
        -----
        1. Close small gaps (dilation then erosion).
        2. Remove salt-and-pepper noise (opening = erosion then dilation).
        3. Keep only the single largest connected component (the lesion).
        4. Flood-fill internal holes.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Largest connected component only
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels > 1:
            # stats[0] is the background; find the largest foreground blob
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)

        # Fill internal holes via flood-fill from corner
        h, w = mask.shape
        flood = mask.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        mask = cv2.bitwise_or(mask, holes)

        return mask

    # ------------------------------------------------------------------
    # Public helper – coloured overlay for UI display
    # ------------------------------------------------------------------

    def get_overlay(self, alpha: float = 0.35) -> np.ndarray:
        """
        Return the source image with the mask region highlighted in cyan.

        Parameters
        ----------
        alpha : float
            Opacity of the cyan highlight in [0, 1].
        """
        overlay = self._source.copy()
        if self._result is None:
            return overlay

        # Cyan highlight on the masked region
        highlight = overlay.copy()
        highlight[self._result > 0] = [255, 230, 0]   # BGR cyan-yellow
        cv2.addWeighted(highlight, alpha, overlay, 1 - alpha, 0, overlay)

        # Draw contour of the mask boundary
        contours, _ = cv2.findContours(
            self._result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 200, 255), 2)

        return overlay
