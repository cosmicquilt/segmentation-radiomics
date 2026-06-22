"""does the texture result converge as the lidc cohort grows, or is it genuinely low-signal?

builds the lidc 4-rater cohort once, extracts features per radiologist mask, then subsamples at
increasing cohort sizes and recomputes the inter-observer reproducibility at each. charts (top)
median icc(2,1) per family and (bottom) the count of texture features still flagged low-signal,
both vs the number of nodules, so the small-subset "texture is ill-conditioned" caveat can be
tested directly: if the low-signal count falls as the cohort grows, texture was just underpowered;
if it stays flat, texture is genuinely near-constant here.

needs pylidc + a lidc download (set LIDC_DICOM_ROOT, and LIDC_LIMIT for how many scans to load).

    python scripts/cohort_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seg_radiomics.data.lidc import build_lidc_cohort  # noqa: E402
from seg_radiomics.features import FEATURE_NAMES, extract_features  # noqa: E402
from seg_radiomics.reproducibility import feature_class, icc_2_1  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
N_RATERS = 4
MIN_SNR = 3.0   # same low-signal threshold as the pipeline
REPEATS = 5     # random subsamples averaged per cohort size, to smooth the curve


def family_stats(rows: list[dict], idx) -> tuple[dict, int]:
    """per-family median inter-observer icc + count of low-signal texture features over idx"""
    by_fam = {"shape": [], "firstorder": [], "texture": []}
    n_low_texture = 0
    for name in FEATURE_NAMES:
        mat = np.array([rows[i][name] for i in idx], dtype=float)  # (n_nodules, n_raters)
        ok = np.isfinite(mat).all(axis=1)
        if ok.sum() < 3:
            continue
        mat = mat[ok]
        between = mat[:, 0].std()
        within = (mat - mat.mean(axis=1, keepdims=True)).std()
        fam = feature_class(name)
        if fam in by_fam:
            by_fam[fam].append(icc_2_1(mat))
        if fam == "texture" and between / (within + 1e-12) < MIN_SNR:
            n_low_texture += 1
    meds = {f: (float(np.nanmedian(v)) if v else float("nan")) for f, v in by_fam.items()}
    return meds, n_low_texture


def main() -> None:
    root = os.environ.get("LIDC_DICOM_ROOT", "/content/drive/MyDrive/lidc")
    limit = int(os.environ.get("LIDC_LIMIT", "200"))
    cohort = build_lidc_cohort(limit=limit, dicom_root=root)
    quad = [c for c in cohort if len(c.get("rater_masks") or []) >= N_RATERS]
    print(f"4-rater nodules available: {len(quad)}")
    if len(quad) < 20:
        print("too few 4-rater nodules for a sweep, download more series")
        return

    # extract every feature for every radiologist mask once, then subsample the table
    rows = []
    for c in quad:
        img = c["image"]
        fs = [extract_features(img, np.asarray(m) > 0) for m in c["rater_masks"][:N_RATERS]]
        rows.append({k: [f[k] for f in fs] for k in FEATURE_NAMES})

    rng = np.random.default_rng(0)
    sizes = [s for s in (20, 40, 60, 80, 120, 160, 200, 300, 400) if s < len(quad)] + [len(quad)]
    xs, sh, fo, tx, low = [], [], [], [], []
    print(f"{'n':>5} {'shape':>7} {'firstord':>9} {'texture':>8} {'low-sig tex/10':>15}")
    for n in sizes:
        meds_reps, low_reps = [], []
        for _ in range(REPEATS):
            idx = (np.arange(len(quad)) if n >= len(quad)
                   else rng.choice(len(quad), n, replace=False))
            meds, nlow = family_stats(rows, idx)
            meds_reps.append(meds)
            low_reps.append(nlow)
        sh_m = float(np.mean([m["shape"] for m in meds_reps]))
        fo_m = float(np.mean([m["firstorder"] for m in meds_reps]))
        tx_m = float(np.mean([m["texture"] for m in meds_reps]))
        low_m = float(np.mean(low_reps))
        xs.append(n); sh.append(sh_m); fo.append(fo_m); tx.append(tx_m); low.append(low_m)
        print(f"{n:>5} {sh_m:>7.3f} {fo_m:>9.3f} {tx_m:>8.3f} {low_m:>15.1f}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(xs, sh, "o-", color="#D55E00", label="shape")
    a1.plot(xs, fo, "o-", color="#0072B2", label="first-order")
    a1.plot(xs, tx, "o-", color="#009E73", label="texture")
    a1.axhline(0.75, ls="--", color="#888888", lw=1.0)
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("median inter-observer ICC(2,1)")
    a1.legend(fontsize=8, loc="lower right")
    a1.grid(ls=":", color="#dddddd")
    a1.set_title("does texture reproducibility converge as the cohort grows?\n"
                 "lidc 4-radiologist ICC vs number of nodules (raw, pre-floor)", fontsize=11)
    a2.plot(xs, low, "o-", color="#009E73")
    a2.set_ylim(-0.3, 10.3)
    a2.set_ylabel("# low-signal\ntexture feats (/10)")
    a2.set_xlabel("number of 4-rater nodules")
    a2.grid(ls=":", color="#dddddd")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "cohort_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
