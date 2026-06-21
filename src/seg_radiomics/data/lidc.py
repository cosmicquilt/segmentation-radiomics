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
    import configparser

    # pylidc targets old python/numpy and calls several apis removed in modern versions
    # (colab is python 3.12 + numpy 2.x). restore the ones it uses before importing it:
    # configparser.SafeConfigParser (gone in 3.12) and the np.* builtin aliases (gone in
    # numpy 1.24 / 2.0, e.g. np.int in Contour.to_matrix, np.bool in the mask builders)
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser
    # note: skip np.object / np.str, probing them emits a futurewarning and pylidc does not
    # need them (the run gets through cluster_annotations + consensus without them)
    for _name, _builtin in (("int", int), ("float", float), ("bool", bool),
                            ("long", int), ("complex", complex)):
        if not hasattr(np, _name):
            setattr(np, _name, _builtin)
    try:
        import pylidc as pl
        from pylidc.utils import consensus

        return pl, consensus
    except Exception as exc:  # pragma: no cover exercised only without pylidc
        raise RuntimeError(
            "pylidc is required for the LIDC loader. Install it (`pip install pylidc`) "
            "and set data.dicom_root to the tcia download dir (or create ~/.pylidcrc) "
            "(see scripts/download_data.md)."
        ) from exc


def build_lidc_cohort(
    limit: int | None = 50,
    clevel: float = 0.5,
    pad: int = 2,
    malignant_threshold: int = 3,
    exclude_indeterminate: bool = False,
    min_annotations: int = 1,
    dicom_root: str | None = None,
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
    dicom_root flat tcia_utils download dir (<root>/<series_uid>/*.dcm); when set the loader
        points pylidc straight at each scan's series folder, so no ~/.pylidcrc or file reorg is
        needed and only downloaded series load. none falls back to pylidc + ~/.pylidcrc

    returns list of dicts {image mask label malignancy spacing scan_id}
    """
    import os

    pl, consensus = _require_pylidc()

    if dicom_root:
        # tcia_utils saves <root>/<SeriesInstanceUID>/*.dcm (flat), but pylidc expects a
        # patient/study/series hierarchy via ~/.pylidcrc. point each scan straight at its flat
        # series folder so the subset download loads as-is (also sidesteps pylidc's config read)
        pl.Scan.get_path_to_dicom_files = (
            lambda self, *a, **k: os.path.join(dicom_root, self.series_instance_uid)
        )

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
        if dicom_root and not os.path.isdir(os.path.join(dicom_root, scan.series_instance_uid)):
            continue  # series not in the downloaded subset
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
            # consensus also returns the individual radiologist masks (3rd value), all aligned
            # to cbbox, which feed the real inter-observer reproducibility analysis
            cmask, cbbox, rater_masks = consensus(
                annotations, clevel=clevel, pad=[(pad, pad), (pad, pad), (pad, pad)]
            )
            malignancy = float(np.median([a.malignancy for a in annotations]))
            if exclude_indeterminate and malignancy == 3:
                continue
            cohort.append(
                {
                    "image": np.asarray(volume[cbbox], dtype=np.float32),
                    "mask": np.asarray(cmask, dtype=bool),
                    "rater_masks": [np.asarray(m, dtype=bool) for m in rater_masks],
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
