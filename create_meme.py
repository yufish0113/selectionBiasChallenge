"""
Assemble the four-panel statistics meme (Reality, Model, Selection Bias, Estimate).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _ensure_gray_01(arr: np.ndarray) -> np.ndarray:
    """Return a 2D float32 array clipped to [0, 1]."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("Each image must be a 2D (height, width) grayscale array.")
    return np.clip(a, 0.0, 1.0)


def _resize_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Scale grayscale ``img`` to (height, width) with LANCZOS if needed."""
    img = _ensure_gray_01(img)
    if img.shape == (height, width):
        return img
    pil = Image.fromarray((img * 255.0 + 0.5).astype(np.uint8), mode="L")
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Build a 1×4 row of panels with labels and save as a PNG on disk.

    Pixels are written at ``dpi`` resolution. Each subplot shows one grayscale
    image with the same aspect ratio. If inputs differ in size, they are
    resampled to match ``original_img`` (height, width).

    Parameters
    ----------
    original_img : np.ndarray
        Reference image (2D, values in [0, 1]); defines output panel size.
    stipple_img : np.ndarray
        Stippled / blue-noise model image.
    block_letter_img : np.ndarray
        Mask image (e.g. block letter on white).
    masked_stipple_img : np.ndarray
        Stipple image with mask applied (biased estimate).
    output_path : str
        Path to the output ``.png`` file (parent directories are created if needed).
    dpi : int
        Resolution passed to :meth:`Figure.savefig`.
    background_color : str
        Matplotlib color for figure and axes background (e.g. ``"white"``).

    Returns
    -------
    None
        Writes the file to ``output_path`` and closes the figure.
    """
    h, w = original_img.shape[:2]
    panels: list[tuple[np.ndarray, str]] = [
        (_resize_to(original_img, h, w), "Reality"),
        (_resize_to(stipple_img, h, w), "Your Model"),
        (_resize_to(block_letter_img, h, w), "Selection Bias"),
        (_resize_to(masked_stipple_img, h, w), "Estimate"),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16.5, 4.6),
        layout="constrained",
    )
    fig.patch.set_facecolor(background_color)

    for ax, (data, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(background_color)
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            color="#1a1a1a",
            pad=12,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor("#2c2c2c")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        facecolor=background_color,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)
