"""synthetic lung-ct-like volumes so the pipeline runs before any tcia download

each sample is a low-density lung background (~ -800 hu) with a single denser
spherical nodule its binary ground-truth mask and a binary malignancy-like label
the label depends honestly on the nodule size and intensity (plus noise) so the
radiomic features downstream genuinely correlate with it exactly what the
correlation step is meant to surface but on data we control real targets lidc-idri
/ nsclc-radiomics (tcia) or the msd lung task
"""
from __future__ import annotations

import numpy as np


def make_lesion_volume(shape: tuple[int, int, int] = (48, 64, 64), seed: int | None = None) -> dict:
    """one synthetic ct volume with a nodule its mask and a label"""
    rng = np.random.default_rng(seed)
    d, h, w = shape

    background = rng.normal(-800.0, 40.0, size=shape)  # lung parenchyma in hu

    radius = rng.uniform(4.0, 10.0)  # nodule radius in voxels
    margin = int(radius) + 2
    cz = rng.uniform(margin, d - margin)
    cy = rng.uniform(margin, h - margin)
    cx = rng.uniform(margin, w - margin)
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w]
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    mask = dist <= radius

    nodule_mean = rng.uniform(-20.0, 60.0)  # solid nodules are denser
    volume = background.copy()
    volume[mask] = rng.normal(nodule_mean, 25.0, size=int(mask.sum()))

    # honest label bigger + denser -> more likely malignant
    logit = (radius - 7.0) / 2.0 + (nodule_mean - 20.0) / 40.0 + rng.normal(0, 0.4)
    label = int(logit > 0)

    return {
        "image": volume.astype(np.float32),
        "mask": mask,
        "label": label,
        "radius": float(radius),
        "nodule_mean_hu": float(nodule_mean),
    }


def make_cohort(n: int = 40, shape: tuple[int, int, int] = (40, 56, 56), seed: int = 0) -> list[dict]:
    """a small synthetic cohort for smoke tests and the offline demo"""
    return [make_lesion_volume(shape, seed=seed + i) for i in range(n)]
