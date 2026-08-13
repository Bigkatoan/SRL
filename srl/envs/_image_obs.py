"""Shared HWC->CHW image-observation detection, used by any wrapper whose
underlying env may hand back pixel observations in HWC layout (Gymnasium's
native convention) while SRL's CNNEncoder expects CHW."""

from __future__ import annotations

import numpy as np


def maybe_hwc_to_chw(arr: np.ndarray) -> np.ndarray:
    """Transpose HWC image observations to CHW for SRL's CNNEncoder.

    Both Isaac Lab tiled cameras and Gymnasium pixel envs (e.g. CarRacing,
    Atari-style Box2D envs) return (N, H, W, C) for batched envs and
    (H, W, C) for single-env observations. SRL's CNNEncoder expects
    (N, C, H, W) / (C, H, W) respectively.

    Detection: last axis has 1, 3, or 4 channels AND the spatial dimensions
    are substantially larger -- this avoids mis-transposing flat state
    vectors that just happen to have a small last dimension.
    """
    if arr.ndim == 4:  # (N, H, W, C) batched
        n, h, w, c = arr.shape
        if c in (1, 3, 4) and h > c and w > c:
            return arr.transpose(0, 3, 1, 2)  # -> (N, C, H, W)
    elif arr.ndim == 3:  # (H, W, C) single env
        h, w, c = arr.shape
        if c in (1, 3, 4) and h > c and w > c:
            return arr.transpose(2, 0, 1)  # -> (C, H, W)
    return arr
