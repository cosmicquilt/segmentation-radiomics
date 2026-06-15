"""threshold + largest-component baseline segmenter (numpy no training)

nodules/tumours are denser than surrounding lung so a hu threshold isolates a
candidate region keeping the largest connected component drops speckle a real if
naive segmenter gives the learned model (monai unet) a dice floor to beat and lets
the features/correlation stages run today
"""
from __future__ import annotations

import numpy as np

from ..morphology import largest_component


def threshold_segment(
    image: np.ndarray,
    threshold: float = -400.0,
    keep_largest: bool = True,
) -> np.ndarray:
    """binary mask of voxels above threshold hu (optionally largest component)"""
    mask = np.asarray(image) > threshold
    if keep_largest and mask.any():
        mask = largest_component(mask)
    return mask
