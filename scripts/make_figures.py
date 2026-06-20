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
RAW = "#9a9a9a"  # raw (pre-floor) markers


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

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for y, f in zip(ys, feats):
        col = FO if _is_first_order(f) else SH
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
        tick.set_color(FO if _is_first_order(f) else SH)
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
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=8, frameon=False)
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


if __name__ == "__main__":
    main()
