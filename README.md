# cancer-imaging segmentation -> radiomics -> outcome correlation

**why this matters for quantitative imaging.** this is the group's core daily
workflow: take a scan, delineate the region of interest, turn that region into
quantitative features, and relate those features to a clinical outcome. each step
has to be reproducible and quality-controlled, because the features become
biomarkers that inform real decisions. this project is a clean, end-to-end version
of exactly that pipeline.

```
segmentation  ->  quantitative feature extraction  ->  correlate with outcome  ->  qc throughout
 (dice / iou)        (shape, intensity, texture)         (per-feature r, auc)      (flag failures)
```

it's the companion to [project 1](../01-mri-reconstruction/): that one recovers a
*trustworthy image*, this one turns trustworthy images into *trustworthy numbers*.

## what runs today vs the build plan

the pipeline runs **end-to-end right now** on synthetic lung-ct-like volumes with a
numpy threshold baseline, no download, no gpu. the learned pieces plug into the same
orchestration:

| stage | runs today (baseline) | production upgrade |
|-------|----------------------|--------------------|
| segmentation | hu threshold + largest component | **monai / nnu-net** 3d u-net, dice loss (`segmentation/model.py`) |
| features | radiomics-lite (shape + first-order, numpy) | **pyradiomics** full texture families (glcm/glrlm/glszm) |
| correlation | per-feature pearson r + roc auc | same, on a real label (malignancy / survival) |
| qc | empty / leakage / fragmentation checks | same |

**build plan:** (1) load **lidc-idri**, the loader is written (`data/lidc.py`,
`configs/lidc.yaml`) and pulls the consensus nodule mask **plus the radiologist
malignancy rating**, a *real* label (no more manufactured one); (2) train the monai
u-net, report dice/iou vs ground truth; (3) extract pyradiomics features; (4)
correlate them with malignancy (pearson + auc). `configs/lidc.yaml` extracts features
from the gt consensus mask so the biomarker correlation isn't contaminated by
segmentation error. then stop, no radiogenomics (that's scope creep). see
`scripts/download_data.md`.

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

reproduce with `python scripts/smoke_test.py`:

- **segmentation:** dice ~1.0 on synthetic volumes. this is high *by construction*,
  the synthetic nodule is cleanly separable from "lung" by a threshold. it proves the
  metric/segmenter plumbing, **not** clinical difficulty. real lung ct is where dice
  becomes meaningful and the learned u-net earns its keep.
- **feature vs label correlation** (representative, the synthetic label was built to
  depend on nodule size + density, and the analysis recovers that honestly):

  | feature | pearson r | auc |
  |---|---|---|
  | firstorder_Energy | +0.50 | 0.82 |
  | shape_VoxelVolume | +0.52 | 0.79 |
  | shape_EquivalentDiameter | +0.47 | 0.79 |
  | shape_Sphericity | -0.42 | 0.19 |

  the point isn't the numbers (they're synthetic), it's that the
  segmentation -> features -> correlation chain produces honest, interpretable
  associations with qc in the loop.

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
metrics, radiomics-lite features, correlation, and qc all unit-tested, exit-0). next:
load real lung ct and train the monai segmenter (build plan above).
