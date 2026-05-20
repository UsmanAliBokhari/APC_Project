"""
utils/helpers.py
----------------
Shared utility functions used across the application.

"""

from __future__ import annotations
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from models.skin_lesion import SkinLesion


# ===========================================================================
# Image load and scaling helpers
# ===========================================================================

def load_image(path: str) -> np.ndarray | None:
    """
    Loads an image from disk and returns it as an RGB numpy 
    array (H, W, 3).
    """
    pil_img = Image.open(path).convert("RGB")
    # convert to a numpy array (H, W, 3) with values 0-255
    img = np.array(pil_img)
    
    return img

def img_to_photoimage(rgb_image: np.ndarray, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
    """
    Converts an RGB image to a Tkinter compatible PhotoImage,
    scaled to fit within max_size while preserving aspect ratio.
    """
    h, w = rgb_image.shape[:2]
    max_w, max_h = max_size

    # proportional resize if the image exceeds the canvas bounds
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb_image = image_resize(rgb_image, new_w, new_h)

    pil_img = Image.fromarray(rgb_image)
    return ImageTk.PhotoImage(pil_img)

def sample_block(img, scale_x, scale_y, coords):
    """
    Returns the averaged pixel value of the source block mapped 
    to the given output coords.
    """
    pix_y, pix_x = coords
    x = int(pix_x * scale_x)
    y = int(pix_y * scale_y)
    block = img[y : y + int(scale_y), x : x + int(scale_x)]
    return block.mean(axis=(0, 1))

def image_resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resizes an image to (new_w, new_h) by averaging source pixel 
    blocks (area interpolation).
    """
    old_h, old_w = img.shape[:2]
    scale_x, scale_y = old_w / new_w, old_h / new_h

    # generate every (y, x) output coordinate pair
    coords = [(y, x) for y in range(new_h) for x in range(new_w)]

    # map sample_block over every coordinate
    pixels = list(map(lambda c: sample_block(img, scale_x, scale_y, c), coords))

    return np.array(pixels, dtype=img.dtype).reshape(new_h, new_w, img.shape[2])

# ===========================================================================
# Risk / summary helpers 
# ===========================================================================

def risk_summary_text(lesions: list) -> str:
    """
    Build a human-readable summary string for a list of SkinLesion objects.
    """
    # filter: keep only fully analysed lesions
    analysed = list(filter(lambda l: l.risk_level is not None, lesions))

    if not analysed:
        return "No lesions analysed yet."

    total = len(analysed)

    # count occurrences of each risk level
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
    """
    return list(filter(lambda l: l.risk_level == level, lesions))


def average_risk_score(lesions: list) -> float:
    """
    Compute the mean risk score across all analysed lesions.
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
