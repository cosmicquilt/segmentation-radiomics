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


def features_to_table(feature_dicts: list[dict]) -> dict[str, list]:
    """transpose a list of per-subject feature dicts into a column table"""
    if not feature_dicts:
        return {}
    names = feature_dicts[0].keys()
    return {name: [fd[name] for fd in feature_dicts] for name in names}
