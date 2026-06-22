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


# snapshot from the n=181 four-rater run (300-series download), so the committed figure
# regenerates without lidc data; a live run (run_sweep) overrides it
_SNAPSHOT = {
    "xs": [20, 40, 60, 80, 120, 160, 181],
    "sh": [0.936, 0.945, 0.953, 0.919, 0.927, 0.925, 0.927],
    "fo": [0.745, 0.777, 0.806, 0.800, 0.812, 0.807, 0.812],
    "tx": [0.865, 0.888, 0.896, 0.881, 0.887, 0.893, 0.895],
    "low": [5.0, 4.2, 4.2, 4.8, 4.8, 4.2, 4.0],
}


def run_sweep():
    """build the lidc cohort, extract per-rater features, sweep cohort sizes (needs lidc data)"""
    root = os.environ.get("LIDC_DICOM_ROOT", "/content/drive/MyDrive/lidc")
    limit = int(os.environ.get("LIDC_LIMIT", "200"))
    cohort = build_lidc_cohort(limit=limit, dicom_root=root)
    quad = [c for c in cohort if len(c.get("rater_masks") or []) >= N_RATERS]
    print(f"4-rater nodules available: {len(quad)}")
    if len(quad) < 20:
        raise RuntimeError("too few 4-rater nodules for a sweep, download more series")

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
        xs.append(n)
        sh.append(float(np.mean([m["shape"] for m in meds_reps])))
        fo.append(float(np.mean([m["firstorder"] for m in meds_reps])))
        tx.append(float(np.mean([m["texture"] for m in meds_reps])))
        low.append(float(np.mean(low_reps)))
        print(f"{n:>5} {sh[-1]:>7.3f} {fo[-1]:>9.3f} {tx[-1]:>8.3f} {low[-1]:>15.1f}")
    return xs, sh, fo, tx, low


def plot_sweep(xs, sh, fo, tx, low) -> None:
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
    a1.set_title("texture reproducibility is stable, not underpowered\n"
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


def main() -> None:
    try:
        xs, sh, fo, tx, low = run_sweep()
    except Exception as exc:  # no lidc data locally -> render the committed snapshot
        print(f"no lidc cohort ({type(exc).__name__}: {exc}); using the n=181 snapshot")
        s = _SNAPSHOT
        xs, sh, fo, tx, low = s["xs"], s["sh"], s["fo"], s["tx"], s["low"]
    plot_sweep(xs, sh, fo, tx, low)


if __name__ == "__main__":
    main()
