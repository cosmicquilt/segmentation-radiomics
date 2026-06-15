"""unit tests for the numpy core (segmentation metrics correlation features qc)

    pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seg_radiomics import correlation, features, qc, seg_metrics  # noqa: E402
from seg_radiomics.data.synthetic import make_lesion_volume  # noqa: E402
from seg_radiomics.morphology import connected_components  # noqa: E402
from seg_radiomics.segmentation.baseline import threshold_segment  # noqa: E402


def test_dice_iou_values():
    a = np.zeros((20, 20), bool)
    a[0:10, 0:10] = True
    assert seg_metrics.dice(a, a) == pytest.approx(1.0, abs=1e-3)
    assert seg_metrics.iou(a, a) == pytest.approx(1.0, abs=1e-3)

    disjoint = np.zeros((20, 20), bool)
    disjoint[10:, 10:] = True
    assert seg_metrics.dice(a, disjoint) == pytest.approx(0.0, abs=1e-3)

    half = np.zeros((20, 20), bool)
    half[5:15, 0:10] = True
    assert seg_metrics.dice(a, half) == pytest.approx(0.5, abs=1e-3)
    assert seg_metrics.iou(a, half) == pytest.approx(1 / 3, abs=1e-3)


def test_confusion_counts():
    gt = np.array([[1, 1], [0, 0]], bool)
    pred = np.array([[1, 0], [1, 0]], bool)
    c = seg_metrics.confusion(pred, gt)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)


def test_pearson_known():
    x = np.arange(10.0)
    assert correlation.pearson(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-6)
    assert correlation.pearson(x, -x) == pytest.approx(-1.0, abs=1e-6)


def test_auc_known():
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    labels = np.array([0, 0, 0, 1])
    assert correlation.auc(scores, labels) == pytest.approx(1.0, abs=1e-6)
    assert correlation.auc(-scores, labels) == pytest.approx(0.0, abs=1e-6)
    # chance separation -> 0.5
    assert correlation.auc(np.array([1.0, 1.0, 1.0, 1.0]), labels) == pytest.approx(0.5, abs=1e-6)


def test_features_finite_and_volume_monotone():
    image = np.full((20, 24, 24), -800.0, np.float32)
    small = np.zeros_like(image, bool)
    small[8:12, 10:14, 10:14] = True
    big = np.zeros_like(image, bool)
    big[6:14, 8:16, 8:16] = True
    image[big] = 30.0
    f_small = features.extract_features(image, small)
    f_big = features.extract_features(image, big)
    assert all(np.isfinite(v) for v in f_big.values())
    assert f_big["shape_VoxelVolume"] > f_small["shape_VoxelVolume"]


def test_features_empty_mask_raises():
    image = np.zeros((8, 8, 8), np.float32)
    with pytest.raises(ValueError):
        features.extract_features(image, np.zeros_like(image, bool))


def test_threshold_segment_recovers_nodule():
    case = make_lesion_volume((32, 40, 40), seed=3)
    pred = threshold_segment(case["image"], threshold=-400.0)
    assert seg_metrics.dice(pred, case["mask"]) > 0.7


def test_connected_components_counts_blobs():
    m = np.zeros((10, 10), bool)
    m[0:3, 0:3] = True
    m[6:9, 6:9] = True
    _, n = connected_components(m)
    assert n == 2


def test_qc_checks():
    assert not qc.check_mask_nonempty(np.zeros((4, 4), bool)).passed
    full = np.ones((4, 4), bool)
    assert not qc.check_volume_fraction(full, max_fraction=0.5).passed
