"""
Step 4: Render a block letter (default "S") for the selection-bias meme mask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _candidate_font_paths() -> list[Path]:
    """Return likely bold/regular TTF paths for the current OS (for text rendering)."""
    paths: list[Path] = []
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        fonts = windir / "Fonts"
        paths.extend(
            [
                fonts / "arialbd.ttf",
                fonts / "calibrib.ttf",
                fonts / "segoeuib.ttf",
                fonts / "verdanab.ttf",
                fonts / "arial.ttf",
                fonts / "calibri.ttf",
            ]
        )
    else:
        paths.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                Path("/Library/Fonts/Arial Bold.ttf"),
            ]
        )
    return paths


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the first available TrueType font at ``size`` pt, or Pillow's default bitmap font."""
    for p in _candidate_font_paths():
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single letter centered on a white canvas; letter is black.

    Parameters
    ----------
    height, width : int
        Output array shape (rows, cols).
    letter : str
        Character(s) to draw (typically one character).
    font_size_ratio : float
        Letter bounding box should fit within ``width * ratio`` and ``height * ratio``.

    Returns
    -------
    np.ndarray
        2D ``float32`` in ``[0, 1]``, white background (1.0), ink toward 0.0.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not (0.0 < font_size_ratio <= 1.0):
        raise ValueError("font_size_ratio should be in (0, 1]")

    max_w = int(width * font_size_ratio)
    max_h = int(height * font_size_ratio)
    max_w = max(max_w, 1)
    max_h = max(max_h, 1)

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    lo, hi = 1, max(width, height) * 2
    best_size = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(mid)
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h and tw > 0 and th > 0:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1

    font = _load_font(best_size)
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) / 2.0 - bbox[0]
    y = (height - th) / 2.0 - bbox[1]

    # Stroke helps thick sans fonts read as a "block" letter on small canvases.
    draw.text((x, y), letter, font=font, fill=0, stroke_width=1, stroke_fill=0)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)
