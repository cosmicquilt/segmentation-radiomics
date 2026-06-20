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
from .correlation import correlate_features, features_to_table
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
        from .data.lidc import build_lidc_cohort

        return build_lidc_cohort(
            limit=data_cfg.get("limit", 50),
            clevel=data_cfg.get("clevel", 0.5),
            pad=data_cfg.get("pad", 2),
            malignant_threshold=data_cfg.get("malignant_threshold", 3),
            exclude_indeterminate=data_cfg.get("exclude_indeterminate", False),
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
    dices, ious, feats, labels = [], [], [], []

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
        except ValueError:
            report.dropped += 1

    table = features_to_table(feats)
    correlations = correlate_features(table, labels)

    # feature reproducibility under +/-1 voxel segmentation perturbation, the stability
    # companion to the outcome correlation (mirrors project 1's reconstruction analysis)
    from .reproducibility import feature_reproducibility, summarize_reproducibility
    repro_rows = feature_reproducibility(cohort, spacing=default_spacing, use_gt=True)
    reproducibility = summarize_reproducibility(repro_rows)

    results = {
        "source": data_cfg.get("source", "synthetic"),
        "features_from": "ground_truth" if use_gt else "prediction",
        "n_cases": len(cohort),
        "n_kept": len(feats),
        "dice_mean": float(np.mean(dices)) if dices else float("nan"),
        "dice_std": float(np.std(dices)) if dices else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "correlations": correlations,
        "reproducibility": reproducibility,
        "reproducibility_per_feature": repro_rows,
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
    for name, stats in list(results["correlations"].items())[:top_k]:
        lines.append(f"| {name} | {stats['pearson_r']:+.3f} | {stats['auc']:.3f} |")

    repro = results.get("reproducibility", {})
    if repro:
        lines += ["", "feature reproducibility under +/-1 voxel segmentation perturbation:",
                  "| family | median ICC | % ICC>0.85 |", "|---|---|---|"]
        for fam in ("ALL", "shape", "firstorder"):
            if fam in repro:
                s = repro[fam]
                lines.append(f"| {fam} (n={s['n']}) | {s['median_icc']:.3f} | {s['pct_icc_gt_0.85']:.0f}% |")
    return "\n".join(lines)
