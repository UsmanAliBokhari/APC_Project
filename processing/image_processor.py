"""
processing/image_processor.py
------------------------------
Abstract base class for all image processing steps.

Each concrete processor (Enhancer, Segmenter, FeatureExtractor) receives
a source image in its constructor, performs work, and exposes the outcome
through the `result` property. 

Subclasses must implement `_process()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import cv2

class ImageProcessor(ABC):
    """
    Base class for all image processors.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image (uint8, shape H×W×3).
    **kwargs
        Passed through to concrete subclasses.
    """

    def __init__(self, image: np.ndarray, **kwargs) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy ndarray.")
        self._source: np.ndarray = image.copy()
        self._result: np.ndarray | None = None
        self._process(**kwargs)

    @abstractmethod
    def _process(self, **kwargs) -> None:
        """
        Perform the processing and store the output in ``self._result``.
        Called automatically by ``__init__``.
        """

    @property
    def result(self) -> np.ndarray:
        """Return the processed output array."""
        if self._result is None:
            raise RuntimeError(f"{type(self).__name__}._process() did not set _result.")
        return self._result

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        """Convert an RGB image to single-channel grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def to_float(image: np.ndarray) -> np.ndarray:
        """Normalise a uint8 image to float32 in [0, 1]."""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Clip and convert a float image back to uint8."""
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
