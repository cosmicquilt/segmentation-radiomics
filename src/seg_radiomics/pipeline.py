"""end-to-end orchestration cohort -> qc -> segment -> metrics -> features -> correlate

runs on synthetic data (manufactured label) or the real lidc-idri cohort
(radiologist malignancy ratings) selected by data.source threshold baseline
segments today swap in monai for the real run without touching the orchestration

two honesty levers
segmentation dice/iou always reported against ground truth (a failed segment is a
low dice a real data point not silently dropped)
features.use_gt_mask extracts the radiomic features from the ground-truth mask so
the feature->outcome correlation is not contaminated by segmentation error
(recommended for lidc answers do these features track malignancy separately from
how good is my segmenter)
"""
from __future__ import annotations

import logging

import numpy as np

from . import qc as _qc
from .correlation import (
    cluster_bootstrap_auc, correlate_features, features_to_table, partial_correlation, volume_confound)
from .data.synthetic import make_cohort
from .features import extract_features
from .seg_metrics import all_seg_metrics
from .segmentation.baseline import threshold_segment

logger = logging.getLogger("seg_radiomics.pipeline")


def build_cohort(cfg: dict) -> list[dict]:
    """build the case cohort selected by data.source (synthetic | lidc)"""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "synthetic").lower()
    if source == "synthetic":
        return make_cohort(
            n=data_cfg.get("n", 40),
            shape=tuple(data_cfg.get("shape", (40, 56, 56))),
            seed=cfg.get("seed", 0),
        )
    if source == "lidc":
        import os

        from .data.lidc import build_lidc_cohort

        return build_lidc_cohort(
            limit=data_cfg.get("limit", 50),
            clevel=data_cfg.get("clevel", 0.5),
            pad=data_cfg.get("pad", 2),
            malignant_threshold=data_cfg.get("malignant_threshold", 3),
            exclude_indeterminate=data_cfg.get("exclude_indeterminate", False),
            dicom_root=os.environ.get("LIDC_DICOM_ROOT") or data_cfg.get("dicom_root"),
        )
    raise ValueError(f"Unknown data.source {source!r} (use 'synthetic' or 'lidc').")


def run_pipeline(cfg: dict) -> dict:
    """run the full pipeline on the configured cohort return a results dict"""
    data_cfg = cfg.get("data", {})
    seg_cfg = cfg.get("segmentation", {})
    feat_cfg = cfg.get("features", {})
    default_spacing = tuple(data_cfg.get("spacing", (1.0, 1.0, 1.0)))
    use_gt = bool(feat_cfg.get("use_gt_mask", False))

    cohort = build_cohort(cfg)
    report = _qc.QCReport()
    dices, ious, feats, labels, groups, scanners = [], [], [], [], [], []

    for case in cohort:
        image, gt, label = case["image"], case["mask"], case["label"]
        spacing = tuple(case.get("spacing", default_spacing))

        pred = threshold_segment(
            image,
            threshold=seg_cfg.get("threshold", -400.0),
            keep_largest=seg_cfg.get("keep_largest", True),
        )
        # segmentation quality always recorded (a failed segment = low dice)
        sm = all_seg_metrics(pred, gt)
        dices.append(sm["dice"])
        ious.append(sm["iou"])

        # features from the gt mask (honest biomarker analysis) or the predicted
        # mask (end-to-end) qc the mask we actually measure from
        feat_mask = gt if use_gt else pred
        if not _qc.qc_case(image, feat_mask, report):
            continue
        try:
            feats.append(extract_features(image, feat_mask, spacing=spacing))
            labels.append(label)
            groups.append(case.get("scan_id"))
            scanners.append(case.get("scanner"))
        except ValueError:
            report.dropped += 1

    table = features_to_table(feats)
    correlations = correlate_features(table, labels)
    # flag features whose predictivity is just a restatement of lesion size (energy etc.)
    vol_confound = volume_confound(table)
    # does any feature predict malignancy *beyond* lesion size? (the volume-residualized capstone
    # to the size-circularity caveat: partial correlation controlling for roi volume)
    partial_corr = partial_correlation(table, labels)
    # patient-clustered bootstrap ci for the top associations: nodules are not independent
    # within a patient, so resample patients (not nodules) for an honest auc interval
    cluster_ci = {}
    if any(g is not None for g in groups):
        for name in list(correlations)[:3]:
            cluster_ci[name] = list(cluster_bootstrap_auc(table[name], labels, groups))

    # clustering-aware models for the top association: cluster-robust logit (always) + a
    # random-intercept glmm (if statsmodels is installed); and combat scanner harmonization
    cluster_models = combat_report = None
    if any(g is not None for g in groups) and correlations:
        from .stats import cluster_robust_logit, glmm_logit
        top = next(iter(correlations))
        cluster_models = {"feature": top,
                          "cluster_robust": cluster_robust_logit(table[top], labels, groups),
                          "glmm": glmm_logit(table[top], labels, groups)}
    if len({s for s in scanners if s and s != "unknown"}) >= 2:
        from .harmonization import batch_variance_explained, combat
        cols = [k for k in table if np.isfinite(table[k]).all()]
        mat = np.array([[table[k][i] for k in cols] for i in range(len(labels))], dtype=float)
        combat_report = {
            "n_batches": len({s for s in scanners if s}),
            "n_features": len(cols),
            "batch_var_before": batch_variance_explained(mat, scanners),
            "batch_var_after": batch_variance_explained(combat(mat, scanners), scanners),
        }

    # feature reproducibility under +/-1 voxel segmentation perturbation, the stability
    # companion to the outcome correlation (mirrors project 1's reconstruction analysis)
    from .reproducibility import feature_reproducibility, summarize_reproducibility
    repro_rows = feature_reproducibility(cohort, spacing=default_spacing, use_gt=True)
    reproducibility = summarize_reproducibility(repro_rows)
    # rerun after a hounsfield floor excludes leaked air, the computational leakage fix
    hu_floor = feat_cfg.get("hu_floor", -300.0)
    repro_rows_floored = feature_reproducibility(cohort, spacing=default_spacing, use_gt=True, hu_floor=hu_floor)
    reproducibility_floored = summarize_reproducibility(repro_rows_floored)

    # real inter-observer reproducibility when per-rater masks are present (lidc), the gold
    # standard that replaces the +/-1 voxel proxy with the actual radiologist disagreement
    from .reproducibility import interobserver_reproducibility, rater_mask_agreement
    n_raters = feat_cfg.get("n_raters", 4)
    interobserver = interobserver_floored = None
    io_n = 0
    io_dice_raw = io_dice_floored = None
    if any(len(c.get("rater_masks") or []) >= n_raters for c in cohort):
        io_rows, io_n = interobserver_reproducibility(cohort, spacing=default_spacing, n_raters=n_raters)
        io_rows_fl, _ = interobserver_reproducibility(cohort, spacing=default_spacing,
                                                      n_raters=n_raters, hu_floor=hu_floor)
        interobserver = summarize_reproducibility(io_rows, threshold=0.75)
        interobserver_floored = summarize_reproducibility(io_rows_fl, threshold=0.75)
        # degeneracy check: if the floor collapses the contours (dice -> 1.0), the floored icc
        # is tautological (identical masks -> identical features) not a real stability gain
        io_dice_raw, _ = rater_mask_agreement(cohort, n_raters=n_raters)
        io_dice_floored, _ = rater_mask_agreement(cohort, n_raters=n_raters, hu_floor=hu_floor)

    # nodules cluster within patients, surfaced so the univariate stats are read honestly
    patients = {c.get("scan_id") for c in cohort if c.get("scan_id")}

    results = {
        "source": data_cfg.get("source", "synthetic"),
        "features_from": "ground_truth" if use_gt else "prediction",
        "n_cases": len(cohort),
        "n_kept": len(feats),
        "dice_mean": float(np.mean(dices)) if dices else float("nan"),
        "dice_std": float(np.std(dices)) if dices else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "correlations": correlations,
        "correlation_cluster_ci": cluster_ci,
        "cluster_models": cluster_models,
        "combat": combat_report,
        "volume_confound": vol_confound,
        "partial_correlation": partial_corr,
        "reproducibility": reproducibility,
        "reproducibility_per_feature": repro_rows,
        "hu_floor": hu_floor,
        "reproducibility_floored": reproducibility_floored,
        "reproducibility_per_feature_floored": repro_rows_floored,
        "n_patients": len(patients) if patients else None,
        "n_raters": n_raters,
        "interobserver": interobserver,
        "interobserver_floored": interobserver_floored,
        "interobserver_n_nodules": io_n,
        "interobserver_dice_raw": io_dice_raw,
        "interobserver_dice_floored": io_dice_floored,
        "qc": report.summary(),
    }
    logger.info("%s", results["qc"])
    return results


# backwards-compatible alias (the synthetic smoke test and cli use this name)
run_synthetic_pipeline = run_pipeline


def format_results(results: dict, top_k: int = 5) -> str:
    """human-readable summary for the cli"""
    lines = [
        f"source: {results.get('source', 'synthetic')}  "
        f"(features from {results.get('features_from', 'prediction')} mask)",
        f"cases: {results['n_kept']}/{results['n_cases']} kept after QC",
        f"segmentation: Dice {results['dice_mean']:.3f} +/- {results['dice_std']:.3f}, "
        f"IoU {results['iou_mean']:.3f}",
        f"{results['qc']}",
        "",
        f"top {top_k} feature <-> label associations:",
        "| feature | Pearson r | AUC |",
        "|---|---|---|",
    ]
    if results.get("n_patients"):
        lines.insert(2, f"clustering: {results['n_kept']} nodules from {results['n_patients']} patients "
                     "(not independent, the univariate stats below are not cluster-corrected)")
    for name, stats in list(results["correlations"].items())[:top_k]:
        lines.append(f"| {name} | {stats['pearson_r']:+.3f} | {stats['auc']:.3f} |")
    cci = results.get("correlation_cluster_ci") or {}
    if cci:
        lines.append("patient-clustered bootstrap 95% CI for AUC (resampling patients, not nodules):")
        for name, (lo, hi) in cci.items():
            lines.append(f"  {name}: [{lo:.3f}, {hi:.3f}]")
    cm = results.get("cluster_models") or {}
    if cm.get("cluster_robust"):
        cr = cm["cluster_robust"]
        lines.append(f"cluster-robust logistic regression of malignancy on {cm['feature']} "
                     f"(clustered by patient, n={cr['n']} / {cr['n_clusters']} patients): "
                     f"OR/SD {cr['odds_ratio_per_sd']:.2f} [{cr['or_ci'][0]:.2f}, {cr['or_ci'][1]:.2f}], p={cr['p']:.3f}")
        if cm.get("glmm"):
            gl = cm["glmm"]
            lines.append(f"  random-intercept GLMM agrees: OR/SD {gl['odds_ratio_per_sd']:.2f} "
                         f"[{gl['or_ci'][0]:.2f}, {gl['or_ci'][1]:.2f}]")
    cb = results.get("combat")
    if cb:
        lines.append(f"combat harmonization across {cb['n_batches']} scanner batches "
                     f"({cb['n_features']} finite features): median batch-variance-explained "
                     f"{cb['batch_var_before']:.3f} -> {cb['batch_var_after']:.3f}")

    repro = results.get("reproducibility", {})
    floored = results.get("reproducibility_floored", {})
    if repro:
        hf = results.get("hu_floor", -300.0)
        lines += ["", f"feature reproducibility under +/-1 voxel perturbation (raw -> {hf:.0f} HU floor):",
                  "| family | median ICC raw | median ICC floored | % ICC>0.85 (raw -> floored) |",
                  "|---|---|---|---|"]
        for fam in ("ALL", "shape", "firstorder", "texture"):
            if fam in repro:
                s, f = repro[fam], floored.get(fam, {})
                fm, fp = f.get("median_icc", float("nan")), f.get("pct_pass", float("nan"))
                lines.append(f"| {fam} (n={s['n']}) | {s['median_icc']:.3f} | {fm:.3f} | "
                             f"{s['pct_pass']:.0f}% -> {fp:.0f}% |")
        low = [r["feature"] for r in results.get("reproducibility_per_feature_floored", [])
               if r.get("low_signal")]
        if low:
            lines.append(f"low-signal (icc ill-conditioned, near-constant across cases): {', '.join(low)}")

    io = results.get("interobserver") or {}
    io_fl = results.get("interobserver_floored") or {}
    if io:
        n, k = results.get("interobserver_n_nodules", 0), results.get("n_raters", 4)
        hf = results.get("hu_floor", -300.0)
        lines += ["", f"REAL inter-observer reproducibility: ICC(2,1) across {k} radiologist masks "
                  f"(n={n} nodules drawn by all {k}, raw -> {hf:.0f} HU floor):",
                  "| family | median ICC raw | median ICC floored | % ICC>0.75 (raw -> floored) |",
                  "|---|---|---|---|"]
        for fam in ("ALL", "shape", "firstorder", "texture"):
            if fam in io:
                s, f = io[fam], io_fl.get(fam, {})
                fm, fp = f.get("median_icc", float("nan")), f.get("pct_pass", float("nan"))
                lines.append(f"| {fam} (n={s['n']}) | {s['median_icc']:.3f} | {fm:.3f} | "
                             f"{s['pct_pass']:.0f}% -> {fp:.0f}% |")
        dr, df = results.get("interobserver_dice_raw"), results.get("interobserver_dice_floored")
        if dr is not None:
            verdict = ("floor collapses the contours, the floored ICC is largely tautological"
                       if (df or 0) >= 0.97 else
                       "contours stay distinct, the floored ICC is a real stability gain")
            lines.append(f"rater mask agreement (mean pairwise Dice across {k}): "
                         f"raw {dr:.3f} -> floored {df:.3f}  ({verdict})")

    vc = results.get("volume_confound", {})
    flagged = [n for n, st in vc.items() if st["volume_proxy"]]
    if flagged:
        lines += ["", "volume-confounded features (|spearman with VoxelVolume| >= 0.7, predictivity is a size proxy):"]
        for n in flagged[:6]:
            lines.append(f"  {n}: rho={vc[n]['spearman_vol']:+.3f}")

    pc = results.get("partial_correlation") or {}
    if pc:
        lines += ["", "size-independent signal (partial r with malignancy, controlling for ROI volume):"]
        for name, st in list(pc.items())[:4]:
            lines.append(f"  {name}: partial r={st['partial_r']:+.3f} (raw r={st['raw_r']:+.3f})")
        best = max(abs(st["partial_r"]) for st in pc.values())
        lines.append("  -> no feature predicts malignancy beyond lesion size"
                     if best < 0.2 else f"  -> strongest size-independent signal |partial r|={best:.2f}")
    return "\n".join(lines)
