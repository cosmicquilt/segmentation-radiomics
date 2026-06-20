# cancer-imaging radiomics: feature reproducibility + outcome correlation

**why this matters for quantitative imaging.** a radiomic feature is only a biomarker
if it does two things: track the clinical outcome, *and* stay stable under the upstream
variability that produced it. a feature that swings whenever the segmentation boundary
moves by a voxel is noise dressed up as a number, no matter how well it correlates. so
this project measures both halves: which features actually associate with the label, and
which of those survive a perturbation of the segmentation.

```
                                    -> feature reproducibility   (ICC / CCC under +/-1 voxel mask perturbation)
segmentation -> feature extraction -|                                                  (which features are trustworthy)
 (dice / iou)   (shape, intensity)  -> outcome correlation       (per-feature pearson r, auc)
                                                       qc throughout (flag failures, never silently average)
```

**this is the downstream half of a two-project pipeline.**
[project 1](../01-mri-reconstruction/) characterized one upstream source of radiomic
feature instability, the image *reconstruction* (it found that the model winning SSIM
was not the model preserving biomarkers). this project characterizes the other upstream
source, the *segmentation*, with the **same ICC / CCC machinery**. project 1 asks how
reconstruction perturbs features, this asks how segmentation does. together they recover
a *trustworthy image* and then turn it into *trustworthy numbers*.

## what runs today vs the build plan

the pipeline runs **end-to-end right now** on synthetic lung-ct-like volumes with a
numpy threshold baseline, no download, no gpu. the learned pieces plug into the same
orchestration:

| stage | runs today (baseline) | production upgrade |
|-------|----------------------|--------------------|
| segmentation | hu threshold + largest component | **monai / nnu-net** 3d u-net, dice loss (`segmentation/model.py`) |
| features | radiomics-lite (shape + first-order, numpy) | **pyradiomics** texture families (glcm/glrlm/glszm), fixed 25 hu bin width, ibsi-compliant |
| reproducibility | icc(2,1) under +/-1 voxel erode/dilate, raw + parenchyma-floored | **real inter-observer** spread (lidc's 4 radiologist masks/nodule), stochastic contour perturbation |
| confound checks | spearman vs roi volume (size-proxy flag) | + small-nodule icc stratification, combat cross-batch harmonization |
| correlation | per-feature pearson r + roc auc | same, on a real label (malignancy / survival) |
| qc | empty / leakage / fragmentation checks | same |

**build plan:** (1) load **lidc-idri**, the loader is written (`data/lidc.py`,
`configs/lidc.yaml`) and pulls the consensus nodule mask **plus the radiologist
malignancy rating**, a *real* label (no more manufactured one); (2) train the monai
u-net, report dice/iou vs ground truth; (3) extract pyradiomics features; (4) correlate
them with malignancy (pearson + auc) **and** rerun the reproducibility analysis on
lidc's **four radiologist annotations per nodule**, real inter-observer variability that
replaces the synthetic erode/dilate proxy with the genuine thing. `configs/lidc.yaml`
extracts the correlation features from the gt consensus mask so the biomarker
correlation isn't contaminated by segmentation error. then stop, no radiogenomics
(that's scope creep). see `scripts/download_data.md`.

the lidc step also picks up the methods upgrades a radiomics review flagged: pyradiomics
texture families (glcm/glrlm) under the 4-rater variance, a **fixed 25 hu bin width** (not
a fixed bin count, so a gray level means the same thing across patients) for ibsi
compliance, a stochastic contour perturbation alongside the deterministic erode/dilate,
icc stratified by nodule size (a one-voxel shift erases more of a 4 mm nodule than a 30 mm
one), and combat / nested-combat as the cross-scanner harmonization step once features
span acquisition batches.

## quickstart

```bash
# core check, numpy only, no download, ~2s
python scripts/smoke_test.py

# install (python 3.10/3.11 recommended)
pip install -r requirements.txt && pip install -e .

# full synthetic pipeline end-to-end
python -m seg_radiomics.cli run --config configs/default.yaml
```

colab: open [`notebooks/colab_segmentation_radiomics.ipynb`](notebooks/colab_segmentation_radiomics.ipynb).
it runs the synthetic pipeline immediately, a gpu runtime (t4/l4/a100) is recommended
once you add the monai segmenter.

## results (synthetic, validates the plumbing, labelled as such)

reproduce both tables with `python -m seg_radiomics.cli run --config configs/default.yaml`
(numpy only, no download, ~3s). the synthetic label was built to depend on nodule size +
density, so the analysis has a real signal to recover. **the numbers are synthetic, the
methodology is the point.**

**segmentation:** dice ~1.0, high *by construction* (the synthetic nodule is cleanly
threshold-separable from "lung"). it proves the metric/segmenter plumbing, **not**
clinical difficulty. real lung ct is where dice becomes meaningful and the learned u-net
earns its keep.

**1. feature reproducibility under +/-1 voxel segmentation perturbation.** each mask is
eroded and dilated by one voxel (an inter-observer-boundary proxy), features are
re-extracted from all three masks, and each feature gets an icc(2,1) across the three
segmentations (two-way random, absolute agreement, the form that generalizes to unseen
raters; lin's ccc is reported alongside as a cross-check). raw, the features split sharply:

| feature | family | ICC | reproducible? |
|---|---|---|---|
| firstorder_90Percentile | first-order | 0.98 | yes |
| firstorder_Maximum | first-order | 0.95 | yes |
| shape_SurfaceArea | shape | 0.89 | yes |
| shape_EquivalentDiameter | shape | 0.86 | yes |
| shape_VoxelVolume | shape | 0.82 | borderline |
| firstorder_Energy | first-order | 0.005 | **no** |
| firstorder_Mean / Entropy / StdDev / Minimum | first-order | ~0.00 | **no** |

the mechanism is real and lung-ct-specific: a +1 voxel dilation leaks into -800 hu lung
air, which dominates the sum/mean-based first-order features (energy squares every voxel,
so one rim of -800 hu air swamps it) but barely touches the *upper* percentiles (90th,
max, anchored by the dense core) or the shape features.

**2. the leakage is fixable, not intrinsic.** the instability above is a
segmentation-contamination artifact, so rerunning the exact same analysis after a -300 hu
floor (standard lung-ct parenchyma exclusion: drop every voxel below -300 hu inside the
mask, removing the air a dilation leaked into) recovers it almost completely:

| family | median ICC raw | median ICC floored | % ICC > 0.85 (raw -> floored) |
|---|---|---|---|
| all 12 | 0.39 | 0.95 | 33% -> 92% |
| shape | 0.84 | 0.94 | 50% -> 100% |
| first-order | 0.01 | 0.97 | 25% -> 88% |

every collapsed first-order feature comes back above 0.85 (energy 0.005 -> 0.95, mean
0.03 -> 1.00; stddev recovers to 0.85 but is flagged low-signal, see the caveat below).
this is the actionable half: the pipeline doesn't just flag unstable features, it shows a
one-line preprocessing fix that restores them (the floor is `features.hu_floor` in the
config).

![raw vs floored ICC per feature: every first-order feature that collapsed under the raw +/-1 voxel perturbation (open circles near zero) recovers above the 0.85 good-reliability line once a -300 HU floor excludes the leaked air (filled circles), while the already-robust upper-percentile features barely move and the shape features recover modestly](docs/figures/parenchyma_floor_recovery.png)

*regenerate with `python scripts/make_figures.py` (numpy + matplotlib, ~3s, fixed seed).*

**3. feature vs label correlation, and the volume confound.** per-feature association with
the synthetic malignancy-like label, plus each feature's spearman correlation with roi
volume (a feature that is "predictive" only because it restates lesion size is not a
biomarker):

| feature | pearson r | auc | spearman w/ volume |
|---|---|---|---|
| shape_EquivalentDiameter | +0.77 | 0.93 | +1.00 (definitional) |
| shape_SurfaceArea | +0.75 | 0.93 | +1.00 (definitional) |
| shape_Sphericity | -0.73 | 0.10 | -0.98 (definitional) |
| firstorder_Energy | +0.72 | **0.99** | **+0.83 (size proxy)** |

shape features correlate with volume *by construction*, so the flag only bites for
intensity features, and exactly one trips it: firstorder_Energy.

**the three analyses together are the whole point.** firstorder_Energy is the single best
label separator (auc 0.99), and it fails for two independent reasons the correlation table
alone would never surface: its predictivity is largely a size proxy (spearman 0.83 with
volume), and its raw reproducibility is the worst of any feature (icc 0.005) until the
parenchyma floor rescues it. a feature has to clear all three bars (associated, not a mere
size proxy, reproducible) to be a real biomarker, which is exactly the upstream->downstream
stability question project 1 asked of reconstruction, asked here of segmentation.

**honest about the proxy.** the +/-1 voxel erode/dilate is a deterministic stand-in for
stochastic inter-observer variability, so the shape-feature ICCs partly reflect grid
geometry (a uniform dilation changes volume by a fixed function) rather than reader
disagreement. the real measurement is lidc's four independent radiologist contours per
nodule (build plan), which the same icc(2,1) code consumes directly. icc is reported per
feature, not as a single family average, because the families are bimodal (the first-order
split above would otherwise vanish into a meaningless "moderate" mean).

one feature, stddev, is flagged **low-signal** (the asterisk in the figure): every
synthetic nodule is built with the same internal noise level, so stddev barely varies
across cases (range 23 to 26) and its icc, recovered or not, has almost nothing to be
reproducible *about*. the pipeline flags it with a heuristic guard (between-case spread
under `min_snr` times the within-case spread, `reproducibility.feature_reproducibility`),
the same kind of degeneracy check project 1 needed when a normalization choice made
first-order features near-constant. real lung ct, where nodule heterogeneity genuinely
varies, gives stddev a real signal and a trustworthy icc.

## quality control (first-class, `qc.py`)

each case is checked before its features enter the table, every drop logged with a
reason:

- **empty mask**, the segmenter found nothing
- **volume fraction**, the mask leaked and filled the volume
- **connected components**, a single-lesion mask fragmented into many pieces
  (warned, not dropped)

this is the robustness/harmonization mindset a quantitative-imaging group needs: a
failed segmentation should be *flagged*, never silently averaged into a biomarker.

## repo layout

```
src/seg_radiomics/
├── seg_metrics.py      # dice, iou, confusion, sensitivity/precision
├── morphology.py       # numpy erosion / surface / connected components
├── features.py         # radiomics-lite (pyradiomics-compatible names)
├── correlation.py      # pearson r + auc + spearman volume-confound check
├── reproducibility.py  # feature icc(2,1) under mask perturbation, raw + parenchyma-floored
├── qc.py               # case-level quality control + report
├── pipeline.py         # cohort -> qc -> segment -> metrics -> features -> correlate
├── cli.py
├── data/
│   ├── synthetic.py    # synthetic lung-ct cohort with an honest label
│   └── lidc.py         # real lidc-idri loader (masks + malignancy via pylidc)
└── segmentation/
    ├── baseline.py     # threshold + largest-component (runs now)
    └── model.py        # monai u-net (production stub)
scripts/   smoke_test.py · download_data.md
configs/   default.yaml · lidc.yaml
tests/     test_core.py
```

## status

pipeline scaffolded and **verified end-to-end on synthetic data** (segmentation
metrics, radiomics-lite features, feature reproducibility, outcome correlation, and qc
all unit-tested, exit-0). next: load real lung ct, train the monai segmenter, and rerun
the reproducibility analysis on lidc's four-radiologist annotations (build plan above).
