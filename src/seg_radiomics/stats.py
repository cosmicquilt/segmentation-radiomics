"""clustering-aware inference for the feature -> outcome association

nodules cluster within patients, so a naive logistic regression (or pearson / auc) treating
nodules as independent understates the uncertainty. two ways to account for it, layered:

- cluster_robust_logit: logistic regression with cluster-robust (sandwich) standard errors
  clustered by patient. numpy only, always available, the frequentist workhorse.
- glmm_logit: a random-intercept logistic mixed model y ~ x + (1|patient) via statsmodels, the
  textbook GLMM. gated, runs only where statsmodels is installed (colab), else returns None.
"""
from __future__ import annotations

import numpy as np


def cluster_robust_logit(x, y, groups, max_iter: int = 100):
    """logistic regression of binary y on x with cluster-robust se clustered by groups

    fits y ~ 1 + x (x standardized) by newton-raphson, then the cluster-robust sandwich
    covariance so the slope's inference is honest under within-cluster correlation. returns the
    slope (log-odds per 1 sd of x), cluster-robust se, z, two-sided p, odds ratio + 95% ci, n,
    and the cluster count, or None if degenerate
    """
    from math import erf, sqrt

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 3 or x.std() < 1e-12:
        return None
    xs = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones_like(xs), xs])

    beta = np.zeros(2)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-9, None)
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(2), X.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break

    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    W = np.clip(p * (1 - p), 1e-9, None)
    bread = np.linalg.inv((X * W[:, None]).T @ X + 1e-8 * np.eye(2))
    scores = (y - p)[:, None] * X  # per-observation score contributions
    meat = np.zeros((2, 2))
    for g in np.unique(groups):
        s = scores[groups == g].sum(axis=0)
        meat += np.outer(s, s)
    cov = bread @ meat @ bread
    slope, se = float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
    z = slope / (se + 1e-12)
    pval = float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))
    return {
        "slope": slope, "se": se, "z": float(z), "p": pval,
        "odds_ratio_per_sd": float(np.exp(slope)),
        "or_ci": [float(np.exp(slope - 1.96 * se)), float(np.exp(slope + 1.96 * se))],
        "n": int(len(y)), "n_clusters": int(len(np.unique(groups))),
    }


def glmm_logit(x, y, groups):
    """random-intercept logistic glmm y ~ x + (1|group) via statsmodels (variational bayes)

    the textbook mixed-effects model for clustered binary data. returns the fixed-effect slope
    (per 1 sd of x) + posterior sd + odds-ratio ci, or None if statsmodels is unavailable or the
    fit fails (so the pipeline degrades gracefully to cluster_robust_logit)
    """
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception:
        return None
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        groups = np.asarray(groups)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, groups = x[ok], y[ok], groups[ok]
        if x.std() < 1e-12 or len(np.unique(groups)) < 3:
            return None
        df = pd.DataFrame({"y": y.astype(int), "x": (x - x.mean()) / x.std(), "g": groups.astype(str)})
        res = BinomialBayesMixedGLM.from_formula("y ~ x", {"g": "0 + C(g)"}, df).fit_vb()
        i = list(res.model.exog_names).index("x")
        coef, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        return {
            "slope": coef, "sd": sd, "odds_ratio_per_sd": float(np.exp(coef)),
            "or_ci": [float(np.exp(coef - 1.96 * sd)), float(np.exp(coef + 1.96 * sd))],
        }
    except Exception:
        return None
