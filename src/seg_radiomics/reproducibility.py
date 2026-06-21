"""radiomic feature reproducibility under segmentation variability

the reproducibility companion to the feature->outcome correlation. a biomarker is only
useful if its feature value is stable when the segmentation boundary moves, so this
perturbs each mask (erode / dilate by one voxel, a +/-1 voxel inter-observer proxy) and
measures how well each feature agrees across the perturbations with lin's ccc and
icc(2,1), aggregated by feature family.

this mirrors project 1's reconstruction-stability analysis (same icc/ccc): project 1
asks how *reconstruction* perturbs radiomic features, this asks how *segmentation* does.
together they characterize the two upstream sources of biomarker instability.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .features import FEATURE_NAMES, extract_features
from .morphology import _shift, erode


def dilate(mask: np.ndarray) -> np.ndarray:
    """6-/4-connectivity dilation add voxels adjacent to the mask along any axis"""
    m = np.asarray(mask) > 0
    out = m.copy()
    for axis in range(m.ndim):
        out |= _shift(m, 1, axis) | _shift(m, -1, axis)
    return out


def feature_class(name: str) -> str:
    """coarse family from a pyradiomics-style feature name"""
    if name.startswith("shape_"):
        return "shape"
    if name.startswith("firstorder_"):
        return "firstorder"
    if name.startswith("glcm_"):
        return "texture"
    return "other"


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """lin's concordance correlation coefficient agreement to the identity line"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 1.0


def icc_2_1(data: np.ndarray) -> float:
    """icc(2,1) two-way random effects absolute agreement single measurement

    data is (n_targets, k_raters): each case is a target, each perturbed mask a rater
    """
    data = np.asarray(data, float)
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = data.mean()
    ss_rows = k * ((data.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(0) - grand) ** 2).sum()
    ss_err = ((data - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1) + 1e-12)
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")


def feature_reproducibility(cohort, spacing=(1.0, 1.0, 1.0), use_gt=True, min_voxels=8,
                            hu_floor=None, min_snr=3.0):
    """per-feature icc / ccc across {reference, eroded, dilated} masks

    the reference mask is the ground truth (use_gt) or the prediction. cases whose mask
    erodes away below min_voxels are skipped so the ratios stay defined.

    hu_floor (when set) intersects every mask with image >= hu_floor before extraction, a
    parenchyma-exclusion step that drops the air voxels a dilation leaked into. this is the
    computational fix for the boundary-leakage failure mode: run with and without it to
    show how much first-order stability the floor recovers.

    a feature whose between-case spread is below min_snr times its within-case (perturbation)
    spread is marked low_signal: it is near-constant across the cohort, so icc has almost no
    variance to be reproducible about and the score is ill-conditioned. common in synthetic
    data where a feature is fixed by construction (e.g. stddev when every nodule shares one
    internal noise level)
    """
    triples: list[tuple[dict, dict, dict]] = []
    for case in cohort:
        ref = np.asarray(case["mask"] if use_gt else case.get("pred", case["mask"])) > 0
        ero, dil = erode(ref), dilate(ref)
        if ref.sum() < min_voxels or ero.sum() < min_voxels:
            continue
        img = case["image"]
        if hu_floor is not None:
            keep = np.asarray(img) >= hu_floor
            ref, ero, dil = ref & keep, ero & keep, dil & keep
            if ref.sum() < min_voxels or ero.sum() < min_voxels:
                continue
        sp = tuple(case.get("spacing", spacing))
        try:
            triples.append((
                extract_features(img, ref, sp),
                extract_features(img, ero, sp),
                extract_features(img, dil, sp),
            ))
        except ValueError:
            continue

    rows = []
    for name in FEATURE_NAMES:
        mat = np.array([[t[0][name], t[1][name], t[2][name]] for t in triples], float)
        ok = np.isfinite(mat).all(axis=1)
        if ok.sum() < 3:
            continue
        mat = mat[ok]
        # icc needs the feature to vary across cases. if the between-case spread is not
        # comfortably larger than the within-case (perturbation) spread, the icc is
        # ill-conditioned, so flag it instead of reporting a misleading agreement score
        between = mat[:, 0].std()
        within = (mat - mat.mean(axis=1, keepdims=True)).std()
        snr = float(between / (within + 1e-12))
        rows.append({
            "feature": name,
            "fclass": feature_class(name),
            "icc": icc_2_1(mat),                      # agreement across the 3 segmentations
            "ccc": lin_ccc(mat[:, 0], mat[:, 1:].mean(1)),  # reference vs mean perturbed
            "snr": snr,                               # between-case / within-case spread
            "low_signal": bool(snr < min_snr),        # near-constant across cases, icc untrustworthy
        })
    return rows


def rater_mask_agreement(cohort, n_raters=4, hu_floor=None, min_voxels=8):
    """mean pairwise dice across each nodule's radiologist masks (raw, or after hu_floor)

    the degeneracy check for the floored inter-observer icc: if the floor collapses the
    n_raters distinct contours onto the same dense core, their dice approaches 1.0, which makes
    a near-perfect floored icc tautological (identical masks -> identical features) rather than
    a real stability gain. returns (mean_dice, n_nodules)
    """
    import itertools

    from .seg_metrics import dice

    vals = []
    for case in cohort:
        masks = case.get("rater_masks")
        if not masks or len(masks) < n_raters:
            continue
        ms = [np.asarray(m) > 0 for m in masks[:n_raters]]
        if hu_floor is not None:
            keep = np.asarray(case["image"]) >= hu_floor
            ms = [m & keep for m in ms]
        if any(m.sum() < min_voxels for m in ms):
            continue
        vals.append(float(np.mean([dice(a, b) for a, b in itertools.combinations(ms, 2)])))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def summarize_reproducibility(rows, threshold: float = 0.85) -> dict:
    """median icc and percent of features with icc > threshold, per family and overall"""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[r["fclass"]].append(r["icc"])
        buckets["ALL"].append(r["icc"])
    out = {}
    for fam, vals in buckets.items():
        arr = np.array(vals, float)
        out[fam] = {
            "n": int(arr.size),
            "median_icc": round(float(np.nanmedian(arr)), 3),
            "pct_pass": round(100.0 * float(np.mean(arr > threshold)), 1),
            "threshold": threshold,
        }
    return out


def interobserver_reproducibility(cohort, spacing=(1.0, 1.0, 1.0), n_raters=4, hu_floor=None,
                                  min_voxels=8, min_snr=3.0):
    """per-feature icc(2,1) across the individual radiologist masks of each nodule

    the real inter-observer counterpart to feature_reproducibility: instead of perturbing one
    mask with erode/dilate, it treats the up to four lidc radiologist annotations as the raters.
    a nodule is used only if it carries at least n_raters masks (a balanced k=n_raters icc) and
    the first n_raters are taken; hu_floor applies the parenchyma floor. each case needs an
    "image" crop and a "rater_masks" list aligned to it (the consensus() per-annotation masks).
    returns (rows, n_nodules_used)
    """
    tuples: list[list[dict]] = []
    for case in cohort:
        masks = case.get("rater_masks")
        if not masks or len(masks) < n_raters:
            continue
        img = np.asarray(case["image"])
        keep = (img >= hu_floor) if hu_floor is not None else None
        sp = tuple(case.get("spacing", spacing))
        feats, ok = [], True
        for m in masks[:n_raters]:
            mm = np.asarray(m) > 0
            if keep is not None:
                mm = mm & keep
            if mm.sum() < min_voxels:
                ok = False
                break
            try:
                feats.append(extract_features(img, mm, sp))
            except ValueError:
                ok = False
                break
        if ok:
            tuples.append(feats)

    rows = []
    for name in FEATURE_NAMES:
        mat = np.array([[f[name] for f in tup] for tup in tuples], float)
        if mat.size == 0:
            continue
        ok = np.isfinite(mat).all(axis=1)
        if ok.sum() < 3:
            continue
        mat = mat[ok]
        between = mat[:, 0].std()
        within = (mat - mat.mean(axis=1, keepdims=True)).std()
        snr = float(between / (within + 1e-12))
        rows.append({
            "feature": name,
            "fclass": feature_class(name),
            "icc": icc_2_1(mat),                    # agreement across the real radiologist masks
            "snr": snr,
            "low_signal": bool(snr < min_snr),
        })
    return rows, len(tuples)
