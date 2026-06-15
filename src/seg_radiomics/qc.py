"""quality control for the segmentation->radiomics pipeline

mirrors project 1 philosophy explicit logged checks that flag failed segmentations
before they silently corrupt the feature table headline check did the segmentation
find a plausible region an empty mask a mask that fills the whole volume or one that
fragments into many pieces all mean the segmenter failed and the features should be
dropped
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .morphology import connected_components

logger = logging.getLogger("seg_radiomics.qc")


@dataclass
class QCResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QCReport:
    results: list[QCResult] = field(default_factory=list)
    dropped: int = 0
    checked: int = 0

    def record(self, result: QCResult, *, drop_on_fail: bool = True) -> bool:
        self.results.append(result)
        if result.passed:
            logger.debug("QC PASS [%s] %s", result.name, result.detail)
            return True
        logger.warning("QC FAIL [%s] %s", result.name, result.detail)
        if drop_on_fail:
            self.dropped += 1
            return False
        return True

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        return (
            f"QC: {passed}/{len(self.results)} checks passed, "
            f"{self.dropped} case(s) dropped of {self.checked} examined."
        )


def check_mask_nonempty(mask: np.ndarray) -> QCResult:
    n = int((np.asarray(mask) > 0).sum())
    return QCResult("mask.nonempty", n > 0, f"voxels={n}")


def check_volume_fraction(mask: np.ndarray, max_fraction: float = 0.5) -> QCResult:
    """a nodule mask shouldnt fill most of the volume that means leakage"""
    m = np.asarray(mask) > 0
    frac = float(m.mean())
    return QCResult("mask.volume_fraction", frac <= max_fraction, f"fraction={frac:.3f}")


def check_single_component(mask: np.ndarray, max_components: int = 1) -> QCResult:
    """a single-lesion segmentation should be one connected blob"""
    _, n = connected_components(np.asarray(mask) > 0)
    return QCResult("mask.components", n <= max_components, f"components={n}")


def check_intensities_finite(image: np.ndarray) -> QCResult:
    ok = bool(np.all(np.isfinite(image)))
    return QCResult("image.finite", ok, "all finite" if ok else "NaN/Inf present")


def qc_case(image: np.ndarray, mask: np.ndarray, report: QCReport, *, strict: bool = True) -> bool:
    """run the standard case-level checks returns true if the case is kept"""
    report.checked += 1
    keep = True
    keep &= report.record(check_intensities_finite(image), drop_on_fail=strict)
    keep &= report.record(check_mask_nonempty(mask), drop_on_fail=strict)
    keep &= report.record(check_volume_fraction(mask), drop_on_fail=strict)
    keep &= report.record(check_single_component(mask), drop_on_fail=False)  # warn only
    return keep
