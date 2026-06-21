"""unit tests for the numpy core (segmentation metrics correlation features qc)

    pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seg_radiomics import correlation, features, qc, reproducibility, seg_metrics  # noqa: E402
from seg_radiomics.data.synthetic import make_cohort, make_lesion_volume  # noqa: E402
from seg_radiomics.morphology import connected_components, erode  # noqa: E402
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
    rng = np.random.default_rng(0)
    image[big] = rng.normal(30.0, 30.0, size=int(big.sum())).astype(np.float32)  # textured, so glcm is defined
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


def test_dilate_erode_change_mask_size():
    m = np.zeros((9, 9, 9), bool)
    m[3:6, 3:6, 3:6] = True
    assert reproducibility.dilate(m).sum() > m.sum()
    assert erode(m).sum() < m.sum()


def test_icc_agreement_and_offset():
    col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    same = np.stack([col, col, col], axis=1)
    assert reproducibility.icc_2_1(same) == pytest.approx(1.0, abs=1e-6)
    # a systematically biased segmentation lowers absolute-agreement icc below perfect
    offset = np.stack([col, col, col + 3.0], axis=1)
    assert reproducibility.icc_2_1(offset) < reproducibility.icc_2_1(same)


def test_lin_ccc_identity_and_offset():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert reproducibility.lin_ccc(x, x) == pytest.approx(1.0, abs=1e-9)
    assert reproducibility.lin_ccc(x, x + 5.0) < 1.0  # offset penalized


def test_feature_reproducibility_structure_and_ranking():
    cohort = make_cohort(n=12, shape=(32, 40, 40), seed=1)
    rows = reproducibility.feature_reproducibility(cohort, use_gt=True)
    assert rows and all(np.isfinite(r["icc"]) for r in rows)
    summary = reproducibility.summarize_reproducibility(rows)
    assert summary["ALL"]["n"] == len(rows)
    # boundary leakage into low-hu air makes mean-based first-order less reproducible
    # than shape (the documented finding), a characterization test on fixed seed
    by = {r["feature"]: r["icc"] for r in rows}
    assert by["shape_SurfaceArea"] > by["firstorder_Mean"]


def test_spearman_captures_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x ** 3  # monotonic but nonlinear, spearman = 1 while pearson < 1
    assert correlation.spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert correlation.pearson(x, y) < 1.0


def test_volume_confound_flags_size_proxy():
    table = {
        "shape_VoxelVolume": [10.0, 20.0, 30.0, 40.0, 50.0],
        "size_proxy": [11.0, 21.0, 29.0, 41.0, 50.0],  # tracks volume
        "independent": [5.0, 1.0, 4.0, 2.0, 3.0],       # unrelated ordering
    }
    vc = correlation.volume_confound(table, flag_threshold=0.7)
    assert vc["size_proxy"]["volume_proxy"]
    assert not vc["independent"]["volume_proxy"]


def test_hu_floor_recovers_first_order_reproducibility():
    cohort = make_cohort(n=12, shape=(32, 40, 40), seed=2)
    raw = {r["feature"]: r["icc"] for r in reproducibility.feature_reproducibility(cohort, use_gt=True)}
    flo = {r["feature"]: r["icc"]
           for r in reproducibility.feature_reproducibility(cohort, use_gt=True, hu_floor=-300.0)}
    # raw, the dilation leaks into -800 hu air and collapses the mean; the floor rescues it
    assert raw["firstorder_Mean"] < 0.5 and flo["firstorder_Mean"] > 0.85


def test_low_signal_guard_flags_near_constant_feature():
    # the canonical cohort the figure uses, where the heuristic is deterministic
    cohort = make_cohort(n=40, shape=(40, 56, 56), seed=0)
    rows = {r["feature"]: r
            for r in reproducibility.feature_reproducibility(cohort, use_gt=True, hu_floor=-300.0)}
    # stddev has by far the lowest between/within signal (nodules share one internal noise
    # level), an order of magnitude under mean (which varies with nodule density)
    assert rows["firstorder_StdDev"]["snr"] < rows["firstorder_Mean"]["snr"] / 10
    # on this cohort the guard flags stddev and not mean; the flag is just snr < min_snr
    assert rows["firstorder_StdDev"]["low_signal"]
    assert not rows["firstorder_Mean"]["low_signal"]


def test_interobserver_reproducibility_structure():
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:24, 0:28, 0:28]
    cohort = []
    for i in range(8):
        rad = 5 + i  # radius varies across nodules -> between-case variance for shape
        base = (zz - 12) ** 2 + (yy - 14) ** 2 + (xx - 14) ** 2 <= rad ** 2
        img = rng.normal(0.0, 1.0, base.shape).astype(np.float32)
        img[base] += 40.0 + i * 4  # intensity varies across nodules too
        # 4 "radiologist" masks = the same nodule drawn with different boundaries
        masks = [base, erode(base), reproducibility.dilate(base), erode(erode(base))]
        cohort.append({"image": img, "mask": base, "rater_masks": masks, "label": i % 2})
    rows, n = reproducibility.interobserver_reproducibility(cohort, n_raters=4)
    by = {r["feature"]: r["icc"] for r in rows}
    assert n == 8 and rows
    # mean intensity varies across nodules and the 4 boundaries agree on it -> finite, high icc
    assert np.isfinite(by["firstorder_Mean"]) and by["firstorder_Mean"] > 0.5
    # a nodule with fewer than n_raters masks is skipped
    _, n2 = reproducibility.interobserver_reproducibility(
        [{"image": cohort[0]["image"], "rater_masks": cohort[0]["rater_masks"][:2]}], n_raters=4)
    assert n2 == 0


def test_rater_mask_agreement_dice():
    img = np.full((16, 20, 20), 100.0, np.float32)  # all >= -300, so the floor is a no-op here
    base = np.zeros((16, 20, 20), bool)
    base[6:12, 8:14, 8:14] = True
    # identical raters -> mean pairwise dice 1.0 (the degenerate case the check exists to catch)
    d_same, n = reproducibility.rater_mask_agreement(
        [{"image": img, "rater_masks": [base, base, base, base]}], n_raters=4)
    assert n == 1 and abs(d_same - 1.0) < 1e-6
    # distinct raters (erode / dilate) -> dice below 1.0
    d_diff, _ = reproducibility.rater_mask_agreement(
        [{"image": img, "rater_masks": [base, erode(base), reproducibility.dilate(base), base]}], n_raters=4)
    assert d_diff < 0.99


def test_glcm_texture_features():
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 50.0, (20, 24, 24)).astype(np.float32)
    mask = np.zeros((20, 24, 24), bool)
    mask[6:14, 8:16, 8:16] = True
    f = features.extract_features(img, mask)
    assert len(features.GLCM_NAMES) == 10
    assert all(np.isfinite(f[k]) for k in features.GLCM_NAMES)
    # a uniform region has < 2 gray levels, so texture is nan (downstream isfinite drops it)
    u = np.full((10, 10, 10), 100.0, np.float32)
    um = np.zeros((10, 10, 10), bool)
    um[3:7, 3:7, 3:7] = True
    fu = features.extract_features(u, um)
    assert all(np.isnan(fu[k]) for k in features.GLCM_NAMES)
