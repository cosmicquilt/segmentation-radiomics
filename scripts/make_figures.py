"""generate the readme hero figure for the reproducibility analysis

renders the parenchyma-floor recovery (raw vs floored icc per feature) to
docs/figures/parenchyma_floor_recovery.png. reproducible: fixed synthetic seed and a
deterministic analysis, so the figure regenerates byte-stably. run:

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from seg_radiomics.data.synthetic import make_cohort  # noqa: E402
from seg_radiomics.reproducibility import feature_reproducibility  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"

FO = "#0072B2"   # first-order family
SH = "#D55E00"   # shape family
TX = "#009E73"   # texture (glcm) family
RAW = "#9a9a9a"  # raw (pre-floor) markers


def _family_color(name: str) -> str:
    if name.startswith("shape_"):
        return SH
    if name.startswith("glcm_"):
        return TX
    return FO


def _short(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def _is_first_order(name: str) -> bool:
    return name.startswith("firstorder_")


def main() -> None:
    cohort = make_cohort(n=40, shape=(40, 56, 56), seed=0)  # matches configs/default.yaml
    raw = {r["feature"]: r["icc"] for r in feature_reproducibility(cohort, use_gt=True)}
    flo_rows = feature_reproducibility(cohort, use_gt=True, hu_floor=-300.0)
    flo = {r["feature"]: r["icc"] for r in flo_rows}
    low = {r["feature"]: r["low_signal"] for r in flo_rows}  # icc untrustworthy, near-constant across cases

    feats = sorted(raw, key=lambda k: raw[k])  # ascending raw icc, the collapsed ones at the bottom
    ys = list(range(len(feats)))

    fig, ax = plt.subplots(figsize=(9.2, max(6.2, 0.42 * len(feats) + 2.2)))
    for y, f in zip(ys, feats):
        col = _family_color(f)
        r0, r1 = raw[f], flo[f]
        if abs(r1 - r0) > 0.02:  # draw the recovery arrow only when it actually moves
            ax.annotate("", xy=(r1, y), xytext=(r0, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, alpha=0.9))
        ax.scatter([r0], [y], s=44, facecolors="white", edgecolors=RAW, linewidths=1.4, zorder=3)
        ax.scatter([r1], [y], s=54, color=col, zorder=4)

    ax.axvline(0.85, ls="--", color="#444", lw=1.0)
    ax.text(0.855, len(feats) - 0.4, "ICC 0.85\ngood reliability\n(Koo & Li)",
            fontsize=8, color="#444", va="top")

    ax.set_yticks(ys)
    ax.set_yticklabels([_short(f) + ("  *" if low.get(f) else "") for f in feats], fontsize=9)
    for tick, f in zip(ax.get_yticklabels(), feats):
        tick.set_color(_family_color(f))
    ax.set_ylim(-0.6, len(feats) - 0.4)
    ax.set_xlim(-0.05, 1.06)
    ax.set_xlabel("ICC(2,1) across the three perturbed segmentations (reference, eroded, dilated)")
    ax.set_title("a -300 HU parenchyma floor recovers the leakage-collapsed features\n"
                 "raw +/-1 voxel perturbation (open) -> air excluded (filled), synthetic lung ct",
                 fontsize=11)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=RAW, markersize=8, label="raw ICC (pre-floor)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FO, markersize=8, label="floored, first-order"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SH, markersize=8, label="floored, shape"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TX, markersize=8, label="floored, texture"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="x", ls=":", color="#dddddd")
    ax.set_axisbelow(True)
    if any(low.values()):
        ax.annotate("*  between-case variance too low for a reliable ICC "
                    "(near-constant across the synthetic cohort)",
                    xy=(0.5, -0.17), xycoords="axes fraction", ha="center", va="top",
                    fontsize=7.5, color="#555555", style="italic")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "parenchyma_floor_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def real_lidc_figure() -> None:
    """proxy vs real-radiologist reproducibility on lidc, by feature family

    regenerates from results/lidc/results.json when a real lidc run is present (colab),
    otherwise uses the snapshot from the n=311 four-rater nodule / 280 patient run
    """
    import json

    def medians(summary):
        return [summary[f]["median_icc"] for f in ("ALL", "shape", "firstorder", "texture")]

    rj = Path(__file__).resolve().parents[1] / "results" / "lidc" / "results.json"
    if rj.exists() and json.loads(rj.read_text()).get("interobserver"):
        r = json.loads(rj.read_text())
        proxy_raw, proxy_flo = medians(r["reproducibility"]), medians(r["reproducibility_floored"])
        real_raw, real_flo = medians(r["interobserver"]), medians(r["interobserver_floored"])
        n_nod, n_pat = r.get("interobserver_n_nodules", "?"), r.get("n_patients", "?")
    else:
        proxy_raw, proxy_flo = [0.519, 0.814, 0.509, 0.439], [0.763, 0.934, 0.815, 0.662]
        real_raw, real_flo = [0.887, 0.922, 0.815, 0.873], [0.982, 0.980, 0.988, 0.982]
        n_nod, n_pat = 311, 280

    fams = ["all 22", "shape", "first-order", "texture *"]
    x = np.arange(len(fams))
    w = 0.2
    series = [
        ("+/-1 voxel proxy, raw", proxy_raw, "#9ecae1"),
        ("+/-1 voxel proxy, floored", proxy_flo, "#3182bd"),
        ("4 radiologists, raw", real_raw, "#fdae6b"),
        ("4 radiologists, floored", real_flo, "#e6550d"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for i, (label, vals, col) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=col)
    for thr, c in ((0.75, "#888888"), (0.85, "#444444")):
        ax.axhline(thr, ls="--", color=c, lw=1.0)
        ax.text(len(fams) - 0.45, thr + 0.005, f"ICC {thr}", fontsize=7.5, color=c, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(fams)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("median ICC(2,1)")
    ax.set_title("real radiologists agree far more than the +/-1 voxel proxy\n"
                 f"lidc inter-observer reproducibility, n={n_nod} nodules / {n_pat} patients",
                 fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", ls=":", color="#dddddd")
    ax.set_axisbelow(True)
    ax.annotate("* texture features are near-constant across this small homogeneous subset (low-signal), so their "
                "ICCs are ill-conditioned, read as indicative only",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
                fontsize=7, color="#555555", style="italic")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "lidc_interobserver.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    real_lidc_figure()
