"""
Step 5: Apply a block-letter mask to the stippled image (selection bias metaphor).
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Where the mask is dark (values below ``threshold``), erase stipples (white).
    Where the mask is light, keep ``stipple_img`` unchanged.

    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image, shape (height, width), values in [0, 1].
    mask_img : np.ndarray
        Mask aligned with stipple; low = masked region (e.g. letter), high = keep.
    threshold : float
        Pixels with mask value strictly below this are treated as mask region.

    Returns
    -------
    np.ndarray
        Same shape as inputs, float32 in [0, 1].
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"Shape mismatch: stipple_img {stipple_img.shape}, mask_img {mask_img.shape}"
        )
    s = np.asarray(stipple_img, dtype=np.float64)
    m = np.asarray(mask_img, dtype=np.float64)
    out = np.where(m < threshold, 1.0, s)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
