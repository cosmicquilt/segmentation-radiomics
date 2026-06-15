"""small numpy morphology helpers (erosion surface connected components)

enough to compute shape features and run qc without scipy/skimage works for 2d and
3d binary masks for production volumes you'd use scipy.ndimage these keep the core
dependency-free and testable
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _shift(a: np.ndarray, s: int, axis: int) -> np.ndarray:
    """shift a by s along axis with false fill (no wraparound)"""
    res = np.zeros_like(a)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if s > 0:
        dst[axis], src[axis] = slice(s, None), slice(None, -s)
    else:
        dst[axis], src[axis] = slice(None, s), slice(-s, None)
    res[tuple(dst)] = a[tuple(src)]
    return res


def erode(mask: np.ndarray) -> np.ndarray:
    """6-/4-connectivity erosion keep voxels whose axis-neighbors are all set"""
    m = np.asarray(mask) > 0
    out = m.copy()
    for axis in range(m.ndim):
        out &= _shift(m, 1, axis) & _shift(m, -1, axis)
    return out


def surface_voxels(mask: np.ndarray) -> int:
    """count boundary voxels (mask minus its erosion) a surface-area proxy"""
    m = np.asarray(mask) > 0
    return int((m & ~erode(m)).sum())


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """label 6-/4-connected components returns (labels count) simple bfs"""
    m = np.asarray(mask) > 0
    labels = np.zeros(m.shape, dtype=np.int32)
    current = 0
    offsets = []
    for axis in range(m.ndim):
        for s in (-1, 1):
            off = [0] * m.ndim
            off[axis] = s
            offsets.append(tuple(off))

    for start in zip(*np.nonzero(m)):
        if labels[start]:
            continue
        current += 1
        queue = deque([start])
        labels[start] = current
        while queue:
            vox = queue.popleft()
            for off in offsets:
                nb = tuple(v + o for v, o in zip(vox, off))
                if all(0 <= nb[i] < m.shape[i] for i in range(m.ndim)) and m[nb] and not labels[nb]:
                    labels[nb] = current
                    queue.append(nb)
    return labels, current


def largest_component(mask: np.ndarray) -> np.ndarray:
    """return the mask of only the largest connected component"""
    labels, n = connected_components(mask)
    if n <= 1:
        return np.asarray(mask) > 0
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    return labels == int(counts.argmax())
