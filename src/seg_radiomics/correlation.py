"""relate radiomic features to a clinical label a simple honest analysis

the jd correlate imaging with clinical/experimental data step deliberately modest
per-feature pearson correlation with the label and a rank-based auc (how well a
single feature separates the two classes) no black-box multivariate model no
p-hacking just transparent univariate associations you can defend pure numpy (no
scipy/sklearn)
"""
from __future__ import annotations

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """pearson correlation coefficient (point-biserial when y is binary)"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / denom) if denom > 0 else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """ranks (1..n) with ties resolved by averaging for a tie-correct auc"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # average tied groups
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """roc auc via the mann-whitney u statistic (tie-aware)"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def correlate_features(feature_table: dict[str, list], labels: list) -> dict:
    """per-feature association with labels

    feature_table maps feature name -> list of values (one per subject)
    returns {feature: {pearson_r:.. auc:.. abs_r:.. n:..}} sorted by descending |r|
    """
    labels = np.asarray(labels).astype(int)
    out = {}
    for name, values in feature_table.items():
        values = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(values)
        if ok.sum() < 3:
            continue
        r = pearson(values[ok], labels[ok])
        out[name] = {
            "pearson_r": r,
            "abs_r": abs(r),
            "auc": auc(values[ok], labels[ok]),
            "n": int(ok.sum()),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["abs_r"], reverse=True))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """spearman rank correlation (pearson on average ranks, tie-aware)"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return pearson(_average_ranks(x), _average_ranks(y))


def volume_confound(feature_table: dict[str, list], volume_key: str = "shape_VoxelVolume",
                    flag_threshold: float = 0.7) -> dict:
    """per-feature spearman with roi volume, flags features that are size proxies

    a feature whose predictive power is mostly a restatement of lesion size is not a
    novel biomarker. energy (sum of squared intensities) scales with voxel count, so it
    tends to flag here, which is exactly why its high auc has to be discounted
    """
    if volume_key not in feature_table:
        return {}
    vol = np.asarray(feature_table[volume_key], dtype=np.float64)
    out = {}
    for name, values in feature_table.items():
        if name == volume_key:
            continue
        v = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(vol)
        if ok.sum() < 3:
            continue
        rho = spearman(v[ok], vol[ok])
        out[name] = {"spearman_vol": rho, "abs": abs(rho), "volume_proxy": bool(abs(rho) >= flag_threshold)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["abs"], reverse=True))


def cluster_bootstrap_auc(scores, labels, groups, n_boot: int = 2000, seed: int = 0):
    """percentile 95% ci for auc that resamples whole patients (groups), not nodules

    the independence-honest interval when several nodules share a patient: a plain bootstrap
    over nodules would understate the ci because intra-patient nodules are correlated. this
    resamples the unique patient ids with replacement, pools their nodules, and recomputes auc.
    returns (lo, hi)
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups)
    ok = np.isfinite(scores)
    scores, labels, groups = scores[ok], labels[ok], groups[ok]
    uniq = np.unique(groups)
    if uniq.size < 3:
        return float("nan"), float("nan")
    members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([members[g] for g in pick])
        a = auc(scores[idx], labels[idx])
        if np.isfinite(a):
            boots.append(a)
    if len(boots) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def partial_correlation(feature_table: dict[str, list], labels: list,
                        control_key: str = "shape_VoxelVolume") -> dict:
    """per-feature partial pearson correlation with labels, controlling for control_key (volume)

    asks the question the size-circularity caveat is really about: does a feature predict the
    outcome *beyond lesion size*? the partial correlation r(f,y|v) = (r_fy - r_fv*r_yv) /
    sqrt((1-r_fv^2)(1-r_yv^2)) removes the size-mediated part, so a feature that only predicted
    through size collapses toward 0. shape features are excluded (they are size/geometry by
    construction), so the candidates are intensity and texture features. returns
    {feature: {partial_r, raw_r, vol_r}} sorted by descending |partial_r|
    """
    if control_key not in feature_table:
        return {}
    y = np.asarray(labels, dtype=np.float64)
    v = np.asarray(feature_table[control_key], dtype=np.float64)
    out = {}
    for name, vals in feature_table.items():
        # shape features are nonlinear size/geometry transforms (diameter ~ volume^1/3), so
        # "size-independent" is malformed for them; the meaningful candidates are intensity/texture
        if name == control_key or name.startswith("shape_"):
            continue
        f = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(f) & np.isfinite(v) & np.isfinite(y)
        if ok.sum() < 4:
            continue
        r_fy, r_fv, r_yv = pearson(f[ok], y[ok]), pearson(f[ok], v[ok]), pearson(y[ok], v[ok])
        if (1 - r_fv**2) < 0.02:
            continue  # feature is essentially volume itself, partial r undefined
        denom = np.sqrt(max((1 - r_fv**2) * (1 - r_yv**2), 1e-12))
        out[name] = {
            "partial_r": float((r_fy - r_fv * r_yv) / denom),
            "raw_r": float(r_fy),
            "vol_r": float(r_fv),
        }
    return dict(sorted(out.items(), key=lambda kv: abs(kv[1]["partial_r"]), reverse=True))


def features_to_table(feature_dicts: list[dict]) -> dict[str, list]:
    """transpose a list of per-subject feature dicts into a column table"""
    if not feature_dicts:
        return {}
    names = feature_dicts[0].keys()
    return {name: [fd[name] for fd in feature_dicts] for name in names}
