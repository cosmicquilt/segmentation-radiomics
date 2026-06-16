"""lidc-idri loader via pylidc real nodule masks and malignancy ratings

the upgrade that replaces the synthetic manufactured label with a real clinical-ish
outcome each nodule carries radiologist malignancy ratings (1-5) which become the
label the radiomic features are correlated against the consensus of the (up to four)
radiologist annotations gives the ground-truth segmentation

requires pylidc and a one-time config ~/.pylidcrc pointing at the extracted lidc
dicom dir e.g.

    [dicom]
    path = /content/drive/MyDrive/lidc/LIDC-IDRI
    warn = True

each case dict matches the synthetic loader contract (image mask label) plus
malignancy and spacing so the rest of the pipeline is unchanged see
scripts/download_data.md
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("seg_radiomics.data.lidc")


def _require_pylidc():
    try:
        import pylidc as pl
        from pylidc.utils import consensus

        return pl, consensus
    except Exception as exc:  # pragma: no cover exercised only without pylidc
        raise RuntimeError(
            "pylidc is required for the LIDC loader. Install it (`pip install pylidc`) "
            "and create ~/.pylidcrc pointing at the extracted LIDC DICOMs "
            "(see scripts/download_data.md)."
        ) from exc


def build_lidc_cohort(
    limit: int | None = 50,
    clevel: float = 0.5,
    pad: int = 2,
    malignant_threshold: int = 3,
    exclude_indeterminate: bool = False,
    min_annotations: int = 1,
) -> list[dict]:
    """load a lidc nodule cohort

    limit cap the number of scans scanned (a subset not the full 133gb) none uses all
    clevel consensus level in [0 1] for merging radiologist annotations
    pad voxels of padding around the consensus bounding box
    malignant_threshold malignancy > this (default 3) -> label 1 (malignant) 3 is the
        radiologists indeterminate
    exclude_indeterminate drop nodules whose median malignancy == 3
    min_annotations require at least this many radiologist annotations per nodule

    returns list of dicts {image mask label malignancy spacing scan_id}
    """
    pl, consensus = _require_pylidc()

    scans = pl.query(pl.Scan)
    total = scans.count()
    logger.info("LIDC: %d scans available", total)

    cohort: list[dict] = []
    for scan in scans[: limit if limit is not None else total]:
        volume = scan.to_volume()  # (rows cols slices) in hu
        spacing = (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_spacing))
        for annotations in scan.cluster_annotations():
            if len(annotations) < min_annotations:
                continue
            cmask, cbbox, _ = consensus(
                annotations, clevel=clevel, pad=[(pad, pad), (pad, pad), (pad, pad)]
            )
            malignancy = float(np.median([a.malignancy for a in annotations]))
            if exclude_indeterminate and malignancy == 3:
                continue
            cohort.append(
                {
                    "image": np.asarray(volume[cbbox], dtype=np.float32),
                    "mask": np.asarray(cmask, dtype=bool),
                    "label": int(malignancy > malignant_threshold),
                    "malignancy": malignancy,
                    "spacing": spacing,
                    "scan_id": scan.patient_id,
                }
            )
    logger.info("LIDC: built cohort of %d nodules", len(cohort))
    return cohort
