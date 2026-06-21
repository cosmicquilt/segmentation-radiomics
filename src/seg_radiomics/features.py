"""radiomics-lite quantitative shape & intensity features from a segmented region

a small dependency-free subset of what pyradiomics extracts: shape, first-order, and a
mask-aware glcm texture family (fixed-bin-width, ibsi-style). enough to drive an honest
features->outcome correlation and a real texture-reproducibility analysis today; the
production path swaps this for pyradiomics for the remaining families (glrlm/glszm/ngtdm)
and fully ibsi-validated definitions. the feature names here mirror pyradiomics so
downstream code is unchanged when you switch
"""
from __future__ import annotations

import numpy as np

from .morphology import surface_voxels

GLCM_BIN_WIDTH = 25.0  # ibsi fixed bin width for ct (hu), so a gray level means the same across patients

GLCM_NAMES = [
    "glcm_Autocorrelation",
    "glcm_ClusterProminence",
    "glcm_ClusterShade",
    "glcm_Contrast",
    "glcm_Correlation",
    "glcm_Dissimilarity",
    "glcm_Idm",
    "glcm_JointEnergy",
    "glcm_JointEntropy",
    "glcm_MaximumProbability",
]

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
] + GLCM_NAMES


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

    feats = {
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
    feats.update(_glcm_features(image, m))
    return feats


def _glcm_offsets(ndim: int) -> list[tuple]:
    """the canonical half of the neighbor directions (first nonzero component positive)"""
    import itertools

    offs = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        if not any(off):
            continue
        for o in off:  # keep only directions whose first nonzero step is +1 (avoid the symmetric dup)
            if o != 0:
                if o > 0:
                    offs.append(off)
                break
    return offs


def _glcm_slices(off: tuple, shape: tuple):
    """aligned source / shifted-neighbor slice tuples for an offset (no wraparound)"""
    a, b = [], []
    for d, s in zip(off, shape):
        lo, hi = max(0, -d), s - max(0, d)
        a.append(slice(lo, hi))
        b.append(slice(lo + d, hi + d))
    return tuple(a), tuple(b)


def _glcm_features(image: np.ndarray, mask: np.ndarray, bin_width: float = GLCM_BIN_WIDTH) -> dict:
    """symmetric, normalized glcm over masked neighbor pairs -> 10 haralick features

    fixed-bin-width discretization on raw hu (ibsi resegmentation style, bin edges at fixed hu
    multiples so a gray level is comparable across patients); nan for every feature when the
    region carries fewer than two gray levels or no in-mask neighbor pairs
    """
    nan = {k: float("nan") for k in GLCM_NAMES}
    v = np.asarray(image, dtype=np.float64)
    disc = np.floor(v / bin_width).astype(np.int64)
    disc = disc - int(disc[mask].min())
    levels = int(disc[mask].max()) + 1
    if levels < 2:
        return nan

    flat = []
    for off in _glcm_offsets(image.ndim):
        sa, sb = _glcm_slices(off, disc.shape)
        both = mask[sa] & mask[sb]
        ii = disc[sa][both].clip(0, levels - 1)
        jj = disc[sb][both].clip(0, levels - 1)
        flat.append(ii * levels + jj)
    if not flat or sum(a.size for a in flat) == 0:
        return nan
    glcm = np.bincount(np.concatenate(flat), minlength=levels * levels).astype(np.float64).reshape(levels, levels)
    glcm = glcm + glcm.T  # symmetric
    glcm /= glcm.sum()

    i_idx, j_idx = np.mgrid[0:levels, 0:levels]
    mu_i, mu_j = (i_idx * glcm).sum(), (j_idx * glcm).sum()
    sig_i = np.sqrt(((i_idx - mu_i) ** 2 * glcm).sum()) + 1e-12
    sig_j = np.sqrt(((j_idx - mu_j) ** 2 * glcm).sum()) + 1e-12
    s = i_idx + j_idx - mu_i - mu_j
    return {
        "glcm_Autocorrelation": float((i_idx * j_idx * glcm).sum()),
        "glcm_ClusterProminence": float(((s ** 4) * glcm).sum()),
        "glcm_ClusterShade": float(((s ** 3) * glcm).sum()),
        "glcm_Contrast": float(((i_idx - j_idx) ** 2 * glcm).sum()),
        "glcm_Correlation": float(((i_idx - mu_i) * (j_idx - mu_j) * glcm).sum() / (sig_i * sig_j)),
        "glcm_Dissimilarity": float((np.abs(i_idx - j_idx) * glcm).sum()),
        "glcm_Idm": float((glcm / (1.0 + (i_idx - j_idx) ** 2)).sum()),
        "glcm_JointEnergy": float((glcm ** 2).sum()),
        "glcm_JointEntropy": float(-(glcm * np.log2(glcm + 1e-12)).sum()),
        "glcm_MaximumProbability": float(glcm.max()),
    }
