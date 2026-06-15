"""end-to-end smoke test for the segmentation->radiomics pipeline (numpy only)

    python scripts/smoke_test.py

checks the metric definitions runs the full synthetic pipeline and asserts the
properties that must hold (segmentation beats chance a size/intensity feature
genuinely tracks the label) exits non-zero on any failure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seg_radiomics import qc, seg_metrics  # noqa: E402
from seg_radiomics.pipeline import format_results, run_synthetic_pipeline  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    section("1. Dice / IoU definitions")
    a = np.zeros((20, 20), bool)
    a[0:10, 0:10] = True
    assert seg_metrics.dice(a, a) > 0.999, "Dice(x,x) must be 1"
    assert seg_metrics.iou(a, a) > 0.999, "IoU(x,x) must be 1"
    disjoint = np.zeros((20, 20), bool)
    disjoint[10:20, 10:20] = True
    assert seg_metrics.dice(a, disjoint) < 1e-3, "disjoint Dice must be ~0"
    half = np.zeros((20, 20), bool)
    half[5:15, 0:10] = True  # overlaps a in rows 5:10 -> 50/100 voxels
    assert abs(seg_metrics.dice(a, half) - 0.5) < 1e-3, "half-overlap Dice must be 0.5"
    assert abs(seg_metrics.iou(a, half) - (1 / 3)) < 1e-3, "half-overlap IoU must be 1/3"
    print("Dice/IoU: identity=1, disjoint=0, half-overlap=0.5/0.333  OK")

    section("2. QC flags an empty mask")
    assert not qc.check_mask_nonempty(np.zeros((8, 8), bool)).passed
    assert qc.check_mask_nonempty(a).passed
    print("empty-mask QC  OK")

    section("3. Full synthetic pipeline")
    results = run_synthetic_pipeline({"seed": 0, "data": {"n": 32, "shape": [36, 52, 52]}})
    print(format_results(results))
    assert results["n_kept"] >= 0.8 * results["n_cases"], "too many cases dropped"
    assert results["dice_mean"] > 0.5, f"threshold segmenter Dice too low: {results['dice_mean']}"

    section("4. A feature genuinely tracks the label")
    top_abs_r = max(stats["abs_r"] for stats in results["correlations"].values())
    top_name = next(iter(results["correlations"]))
    print(f"strongest association: {top_name} (|r|={top_abs_r:.3f})")
    assert top_abs_r > 0.3, "expected a real feature<->label association in synthetic data"

    section("RESULT")
    print("ALL SMOKE-TEST CHECKS PASSED [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
