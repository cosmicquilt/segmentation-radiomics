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

    limit how many scans with dicoms on disk to load (a subset download is normal, scans
        without local dicoms are skipped) none loads every available scan
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
    logger.info("LIDC: %d scans in the pylidc db, loading up to %s that have dicoms on disk",
                scans.count(), limit)

    cohort: list[dict] = []
    used, attempted = 0, 0
    for scan in scans:
        if limit is not None and used >= limit:
            break
        attempted += 1
        # a subset download is normal, so load only scans whose dicoms are actually present
        # rather than assuming the first `limit` scans in db order are the ones that were fetched
        try:
            volume = scan.to_volume(verbose=False)  # (rows cols slices) in hu
        except Exception as exc:
            logger.debug("LIDC: skip %s (no dicoms on disk): %s", scan.patient_id, exc)
            continue
        used += 1
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
    if not cohort:
        raise RuntimeError(
            f"LIDC: loaded 0 scans with dicoms after trying {attempted}. pylidc could not locate "
            "the dicom files. it expects ~/.pylidcrc 'path' to point at the directory that holds "
            "the patient folders, i.e. <path>/LIDC-IDRI-XXXX/<study>/<series>/*.dcm. verify the "
            "download layout matches that (see scripts/download_data.md)."
        )
    logger.info("LIDC: built %d nodules from %d scans (tried %d)", len(cohort), used, attempted)
    return cohort
