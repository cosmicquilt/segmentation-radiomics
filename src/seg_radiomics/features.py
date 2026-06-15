"""radiomics-lite quantitative shape & intensity features from a segmented region

a small dependency-free subset of what pyradiomics extracts enough to drive an
honest features->outcome correlation today production path swaps this for
pyradiomics (radiomics_pyradiomics.py) which adds the full texture families
(glcm/glrlm/glszm) and ibsi-compliant definitions the feature names here mirror
pyradiomics so downstream code is unchanged when you switch
"""
from __future__ import annotations

import numpy as np

from .morphology import surface_voxels

FEATURE_NAMES = [
    "shape_VoxelVolume",
    "shape_SurfaceArea",
    "shape_Sphericity",
    "shape_EquivalentDiameter",
    "firstorder_Mean",
    "firstorder_StdDev",
    "firstorder_Minimum",
    "firstorder_Maximum",
    "firstorder_10Percentile",
    "firstorder_90Percentile",
    "firstorder_Energy",
    "firstorder_Entropy",
]


def extract_features(image: np.ndarray, mask: np.ndarray, spacing: tuple[float, ...] = (1.0, 1.0, 1.0)) -> dict:
    """return a dict of features for the region mask in image

    raises ValueError on an empty mask qc should catch and drop those first
    """
    m = np.asarray(mask) > 0
    if not m.any():
        raise ValueError("empty mask: no region to extract features from")

    spacing = tuple(spacing[: image.ndim]) if len(spacing) >= image.ndim else (1.0,) * image.ndim
    voxel_volume = float(np.prod(spacing))
    face_area = float(np.mean(spacing)) ** 2  # approximate per-voxel surface element

    vals = np.asarray(image)[m].astype(np.float64)
    n = int(m.sum())
    volume = n * voxel_volume
    surface = surface_voxels(m) * face_area
    # sphericity in 3d ratio of a sphere surface (same volume) to actual surface
    if image.ndim == 3 and surface > 0:
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface
    else:
        sphericity = float("nan")
    equiv_diameter = (6 * volume / np.pi) ** (1 / 3) if image.ndim == 3 else (4 * volume / np.pi) ** 0.5

    hist, _ = np.histogram(vals, bins=32)
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    return {
        "shape_VoxelVolume": float(volume),
        "shape_SurfaceArea": float(surface),
        "shape_Sphericity": float(sphericity),
        "shape_EquivalentDiameter": float(equiv_diameter),
        "firstorder_Mean": float(vals.mean()),
        "firstorder_StdDev": float(vals.std()),
        "firstorder_Minimum": float(vals.min()),
        "firstorder_Maximum": float(vals.max()),
        "firstorder_10Percentile": float(np.percentile(vals, 10)),
        "firstorder_90Percentile": float(np.percentile(vals, 90)),
        "firstorder_Energy": float((vals**2).sum()),
        "firstorder_Entropy": entropy,
    }
