"""
utils/helpers.py
----------------
Shared utility functions used across the application.

Demonstrates:
  • list comprehension
  • map()
  • filter()
  • reduce() (from functools)
"""

from __future__ import annotations
from functools import reduce
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from models.skin_lesion import SkinLesion


# ===========================================================================
# Image I/O
# ===========================================================================

def load_image(path: str) -> np.ndarray | None:
    """
    Load an image from *path* as a BGR NumPy array.
    Returns None if the file cannot be read.
    """
    img = cv2.imread(path)
    return img  # None if path invalid


def cv2_to_photoimage(
    bgr_image: np.ndarray,
    max_size: tuple[int, int] = (380, 300),
) -> ImageTk.PhotoImage:
    """
    Convert a BGR NumPy array to a Tkinter-compatible PhotoImage,
    scaled to fit within *max_size* while preserving aspect ratio.

    Steps
    -----
    1. Resize (if needed).
    2. BGR → RGB conversion via cv2.cvtColor.
    3. NumPy → PIL.Image → ImageTk.PhotoImage.
    """
    h, w = bgr_image.shape[:2]
    max_w, max_h = max_size

    # Proportional resize if the image exceeds the canvas bounds
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


# ===========================================================================
# Risk / summary helpers  (map, filter, reduce demos)
# ===========================================================================

def risk_summary_text(lesions: list) -> str:
    """
    Build a human-readable summary string for a list of SkinLesion objects.

    Uses:
      • filter() – select only analysed lesions
      • map()    – extract risk scores
      • reduce() – accumulate counts per level
    """
    # filter: keep only fully analysed lesions
    analysed = list(filter(lambda l: l.risk_level is not None, lesions))

    if not analysed:
        return "No lesions analysed yet."

    total = len(analysed)

    # map + reduce: count occurrences of each risk level
    levels = list(map(lambda l: l.risk_level, analysed))
    counts = reduce(
        lambda acc, lv: {**acc, lv: acc.get(lv, 0) + 1},
        levels,
        {},
    )

    # list comprehension: build "Low ×1, High ×2" style parts
    parts = [f"{lv} ×{cnt}" for lv, cnt in sorted(counts.items())]
    return f"{total} lesion(s) analysed — " + ", ".join(parts)


def filter_by_risk(lesions: list, level: str) -> list:
    """
    Return the subset of *lesions* whose risk_level matches *level*.

    Parameters
    ----------
    lesions : list[SkinLesion]
    level   : "Low" | "Medium" | "High"
    """
    return list(filter(lambda l: l.risk_level == level, lesions))


def average_risk_score(lesions: list) -> float:
    """
    Compute the mean risk score across all analysed lesions.

    Uses map() to extract scores and reduce() to sum them.
    """
    analysed = [l for l in lesions if l.risk_score is not None]
    if not analysed:
        return 0.0

    scores = list(map(lambda l: l.risk_score, analysed))
    total  = reduce(lambda acc, s: acc + s, scores, 0.0)
    return total / len(scores)


def batch_risk_levels(lesions: list) -> list[str]:
    """
    Return a list of risk levels for all analysed lesions.
    Uses a list comprehension with an inline guard.
    """
    return [l.risk_level for l in lesions if l.risk_level is not None]
