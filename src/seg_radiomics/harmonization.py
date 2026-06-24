"""combat harmonization: remove scanner / acquisition-batch effects from radiomic features

parametric empirical-bayes location-and-scale correction (johnson et al. 2007, the method
behind neuroCombat) that realigns each feature's per-batch mean and variance toward a pooled
estimate, so features are comparable across scanners without erasing biological variation.

this is the acquisition-side companion to the segmentation reproducibility analyses: those
characterize feature instability from *contouring*, this corrects the instability from
*acquisition batch* (scanner manufacturer / model). pure numpy, no scipy.
"""
from __future__ import annotations

import numpy as np


def _eb_priors(delta_hat: np.ndarray) -> tuple[float, float]:
    """inverse-gamma prior moments (a, b) from the per-feature variance estimates of a batch"""
    m, s2 = float(delta_hat.mean()), float(delta_hat.var())
    s2 = max(s2, 1e-12)
    a = (2 * s2 + m * m) / s2
    b = (m * s2 + m**3) / s2
    return a, b


def _eb_batch(z: np.ndarray, gamma_hat, delta_hat, gamma_bar, tau2, a, b, tol=1e-4):
    """iterative empirical-bayes gamma* / delta* for one batch (z is n_batch x n_features)"""
    n = z.shape[0]
    g, d = gamma_hat.copy(), delta_hat.copy()
    for _ in range(200):
        g_new = (n * tau2 * gamma_hat + d * gamma_bar) / (n * tau2 + d)
        d_new = (0.5 * ((z - g_new) ** 2).sum(axis=0) + b) / (n / 2.0 + a - 1.0)
        if (np.max(np.abs(g_new - g) / (np.abs(g) + 1e-12)) < tol
                and np.max(np.abs(d_new - d) / (np.abs(d) + 1e-12)) < tol):
            return g_new, d_new
        g, d = g_new, d_new
    return g, d


def combat(features: np.ndarray, batches) -> np.ndarray:
    """harmonize a (n_samples, n_features) array across batch labels (parametric EB combat)

    returns the batch-adjusted array, same shape. batches with < 2 samples are left untouched;
    needs >= 2 usable batches or the input is returned unchanged
    """
    X = np.asarray(features, dtype=np.float64).copy()
    batches = np.asarray(batches)
    n, p = X.shape
    uniq = [bb for bb in np.unique(batches) if int((batches == bb).sum()) >= 2]
    if len(uniq) < 2:
        return X

    # grand mean + pooled within-batch variance per feature, then standardize
    alpha = X.mean(axis=0)
    var_pooled = np.zeros(p)
    for bb in uniq:
        Xb = X[batches == bb]
        var_pooled += ((Xb - Xb.mean(axis=0)) ** 2).sum(axis=0)
    sd = np.sqrt(var_pooled / n) + 1e-8
    Z = (X - alpha) / sd

    # per-batch additive (gamma) and multiplicative (delta) effects, with EB shrinkage
    gamma_hats = np.array([Z[batches == bb].mean(axis=0) for bb in uniq])
    gamma_bar = gamma_hats.mean(axis=0)
    tau2 = gamma_hats.var(axis=0) + 1e-8

    out = X.copy()
    for bb in uniq:
        m = batches == bb
        Zb = Z[m]
        g_hat = Zb.mean(axis=0)
        d_hat = Zb.var(axis=0) + 1e-8
        a, b = _eb_priors(d_hat)
        g_star, d_star = _eb_batch(Zb, g_hat, d_hat, gamma_bar, tau2, a, b)
        out[m] = (Zb - g_star) / np.sqrt(np.maximum(d_star, 1e-8)) * sd + alpha
    return out


def batch_variance_explained(features: np.ndarray, batches) -> float:
    """median over features of the fraction of variance explained by batch (one-way eta^2)

    a simple before/after harmonization metric: combat should drop this toward chance
    """
    X = np.asarray(features, dtype=np.float64)
    batches = np.asarray(batches)
    etas = []
    grand = X.mean(axis=0)
    ss_total = ((X - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(X.shape[1])
    for bb in np.unique(batches):
        Xb = X[batches == bb]
        ss_between += Xb.shape[0] * (Xb.mean(axis=0) - grand) ** 2
    ok = ss_total > 1e-12
    etas = ss_between[ok] / ss_total[ok]
    return float(np.median(etas)) if etas.size else float("nan")


def bimodality_coefficient(x: np.ndarray) -> float:
    """sarle's bimodality coefficient in [0, 1], a dependency-free unimodality screen

    bc = (g1^2 + 1) / (g2 + 3*(n-1)^2 / ((n-2)(n-3))), g1 the sample skewness and g2 the excess
    kurtosis. a normal sits near 0.33, a uniform at 5/9 ~ 0.555 (the conventional flag line), and
    a clean bimodal mixture runs higher. the numpy fallback when the diptest package is absent.
    nan for near-constant or tiny (< 4) samples
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    s = x.std()
    if n < 4 or s < 1e-12:
        return float("nan")
    m = x - x.mean()
    g1 = float((m**3).mean() / s**3)
    g2 = float((m**4).mean() / s**4 - 3.0)
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g1**2 + 1.0) / (g2 + corr))


def unimodality_report(features: np.ndarray, batches, feature_names=None, min_per_batch=8,
                       bc_threshold=5.0 / 9.0) -> dict:
    """flag features whose within-batch distribution is multimodal, breaking combat's assumption

    standard combat applies one additive + one multiplicative shift per batch, which assumes each
    feature is unimodal within a batch. a feature that is bimodal within a batch (mixed
    reconstruction kernels or slice thicknesses, or a solid vs subsolid biological mixture) cannot
    be corrected by moving one mean and one variance, so its harmonization is suspect and gets
    reported rather than trusted silently (the flag-don't-average rule used elsewhere in qc). uses
    hartigan's dip test (the diptest package) for a p-value when installed, otherwise sarle's
    bimodality coefficient. returns {method, n_features, n_suspect, suspect_features, bc_threshold}
    """
    X = np.asarray(features, float)
    batches = np.asarray(batches)
    p = X.shape[1]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(p)]
    try:
        from diptest import diptest as _dip
        method = "hartigan_dip"
    except Exception:
        _dip = None
        method = "bimodality_coefficient"
    uniq = [bb for bb in np.unique(batches) if int((batches == bb).sum()) >= min_per_batch]
    scores: dict[str, float] = {}
    for j in range(p):
        worst = 0.0
        for bb in uniq:
            x = X[batches == bb, j]
            x = x[np.isfinite(x)]
            if x.size < min_per_batch or x.std() < 1e-12:
                continue
            if _dip is not None:
                pval = float(_dip(x)[1])
                if pval < 0.05:                  # reject unimodality
                    worst = max(worst, 1.0 - pval)
            else:
                bc = bimodality_coefficient(x)
                if np.isfinite(bc) and bc > bc_threshold:
                    worst = max(worst, bc)
        if worst > 0.0:
            scores[names[j]] = round(worst, 3)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return {"method": method, "n_features": p, "n_suspect": len(scores),
            "suspect_features": ranked, "bc_threshold": round(bc_threshold, 3)}
