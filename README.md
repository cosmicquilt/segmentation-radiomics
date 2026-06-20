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
| features | radiomics-lite (shape + first-order, numpy) | **pyradiomics** full texture families (glcm/glrlm/glszm) |
| reproducibility | icc / ccc under +/-1 voxel erode/dilate | **real inter-observer** spread (lidc has 4 radiologist masks/nodule) |
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
re-extracted from all three masks, and each feature gets an icc(2,1) + lin's ccc across
the three segmentations. the features split sharply:

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
air, which dominates the sum/mean-based first-order features (energy, mean, entropy, std)
but barely touches the *upper* percentiles (90th, max) or the shape features. this is
precisely why tight, consistent segmentation matters in lung-nodule radiomics. the
*magnitude* here is stark by construction (the synthetic air contrast is extreme), the
*ranking* of which features survive is the transferable result, and lidc's four
radiologist masks per nodule turn this +/-1 voxel proxy into a real inter-observer
measurement (build plan).

**2. feature vs label correlation.** per-feature association with the synthetic
malignancy-like label:

| feature | pearson r | auc |
|---|---|---|
| shape_EquivalentDiameter | +0.77 | 0.93 |
| shape_SurfaceArea | +0.75 | 0.93 |
| shape_VoxelVolume | +0.74 | 0.93 |
| shape_Sphericity | -0.73 | 0.10 |
| firstorder_Energy | +0.72 | **0.99** |

**the two tables together are the whole point.** firstorder_Energy is the single best
label separator (auc 0.99) *and* the least reproducible feature (icc 0.005): correlation
alone would crown it the biomarker, the reproducibility analysis says don't trust it
without a tight segmentation. a feature has to win *both* tables to be a real biomarker,
which is exactly the upstream->downstream stability question project 1 asked of
reconstruction, asked here of segmentation.

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
├── correlation.py      # pearson r + rank-based auc (no scipy/sklearn needed)
├── reproducibility.py  # feature icc / ccc under +/-1 voxel mask perturbation
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
