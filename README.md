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

the pipeline runs **end-to-end right now** two ways: on synthetic lung-ct-like volumes
(numpy, no download, no gpu) **and on real lidc-idri** ct with radiologist masks and
malignancy ratings (needs pylidc + a tcia subset, see the colab notebook). the learned
segmenter is the main piece still to swap in:

| stage | runs today (baseline) | production upgrade |
|-------|----------------------|--------------------|
| segmentation | hu threshold + largest component | **monai / nnu-net** 3d u-net, dice loss (`segmentation/model.py`) |
| features | radiomics-lite (shape + first-order + **glcm texture**, numpy, fixed 25 hu bin width) | **pyradiomics** remaining families (glrlm/glszm/ngtdm), ibsi-validated |
| reproducibility | icc(2,1) under +/-1 voxel erode/dilate **and across lidc's 4 radiologist masks** (real inter-observer), raw + parenchyma-floored | stochastic contour perturbation, texture-family icc |
| confound checks | spearman vs roi volume (size-proxy flag), nodules-per-patient | small-nodule icc stratification, combat cross-batch harmonization |
| correlation | per-feature pearson r + roc auc, synthetic **and real lidc malignancy** | cluster-robust / mixed-effects stats (nodules cluster within patients) |
| qc | empty / leakage / fragmentation checks | same |

**what's done vs left.** the lidc-idri path now runs end-to-end (loader `data/lidc.py`,
`configs/lidc.yaml`): real consensus masks, the radiologist malignancy rating as the label,
the 4-radiologist inter-observer reproducibility, and a **glcm texture family** (fixed 25 hu
bin width), all in the real-data results below. **still on the build plan:** (1) train the
monai u-net to replace the threshold baseline (dice 0.47 on real ct is weak by design); (2) the
remaining **pyradiomics** families (glrlm/glszm/ngtdm) on a texturally *diverse* cohort
(part-solid / ground-glass nodules, not just the solid ones here), since the cohort-size sweep
showed ~half the glcm features are structurally low-variance on this set rather than underpowered;
(3) cluster-robust or mixed-effects stats, since nodules cluster within
patients; (4) validate the malignancy correlation on lidc's ~157-case **pathology-confirmed**
subset, not just the subjective rating. a stochastic contour perturbation, small-nodule icc
stratification, and combat / nested-combat harmonization round out the list. then stop, no
radiogenomics (scope creep).
see `scripts/download_data.md`.

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

## results, part 1: synthetic phantom (the method, illustrated)

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
| all 22 | 0.01 | 0.90 | 18% -> 73% |
| shape | 0.84 | 0.94 | 50% -> 100% |
| first-order | 0.01 | 0.97 | 25% -> 88% |
| texture (glcm) | 0.00 | 0.85 | 0% -> 50% |

every collapsed first-order feature comes back above 0.85 (energy 0.005 -> 0.95, mean
0.03 -> 1.00; stddev recovers to 0.85 but is flagged low-signal, see the caveat below). the
**10 glcm texture features collapse the same way** (raw icc ~0, since the leaked -800 hu air
dominates the co-occurrence matrix) and recover too, but less completely (median 0.85, only
half clear the line) which previews their greater fragility on real readers in part 2. this is
the actionable half: the pipeline doesn't just flag unstable features, it shows a one-line
preprocessing fix that restores them (the floor is `features.hu_floor` in the config).

![raw vs floored ICC for all 22 features: the first-order and glcm texture features that collapsed under the raw +/-1 voxel perturbation (open circles near zero) recover above the 0.85 good-reliability line once a -300 HU floor excludes the leaked air (filled circles), while the already-robust upper-percentile and shape features barely move](docs/figures/parenchyma_floor_recovery.png)

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
nodule, which the same icc(2,1) code consumes directly (see part 2 below, where the
radiologists turn out to agree *far* more than this proxy). icc is reported per feature, not
as a single family average, because the families are bimodal (the first-order split above
would otherwise vanish into a meaningless "moderate" mean).

stddev and several glcm features are flagged **low-signal** (the asterisks in the figure):
every synthetic nodule is built with the same internal noise, so these features barely vary
across the cohort (stddev ranges only 23 to 26) and their icc, recovered or not, has almost
nothing to be reproducible *about*. the pipeline flags any feature whose between-case spread is
under `min_snr` times its within-case spread (`reproducibility.feature_reproducibility`), the
same kind of degeneracy check project 1 needed when a normalization choice made first-order
features near-constant. real lung ct, where nodule texture and heterogeneity genuinely vary,
gives these features a real signal and a trustworthy icc.

## results, part 2: real lidc-idri (the validation)

the identical pipeline runs on **lidc-idri** lung ct (`configs/lidc.yaml`): the segmentation
is the **consensus of up to four radiologist annotations** per nodule, the label is the
radiologists' **malignancy rating (1-5)** binarized at > 3, and the correlation features come
from the consensus mask. one run on a 32-scan subset gave **109 nodules from 32 patients**.

**1. real inter-observer reproducibility (the headline).** instead of the synthetic +/-1
voxel proxy, this treats the **four radiologist masks as four raters** and computes icc(2,1)
across them (n=54 nodules drawn by all four), the gold-standard inter-observer design. the
method holds, and the proxy turns out to have been pessimistic:

| family | median ICC raw | median ICC floored | % ICC > 0.75 (raw -> floored) |
|---|---|---|---|
| all 22 | 0.91 | 0.99 | 77% -> 91% |
| shape | 0.95 | 0.99 | 100% -> 100% |
| first-order | 0.84 | 0.99 | 62% -> 88% |
| texture (glcm) * | 0.89 | 0.99 | 80% -> 90% |

![grouped bars of median ICC by feature family (all, shape, first-order, texture): the four-radiologist inter-observer bars sit far above the +/-1 voxel proxy bars at every family, and the -300 HU floor lifts both to near 1.0; 77 percent of features clear ICC 0.75 raw, shape highest, texture flagged low-signal on this subset](docs/figures/lidc_interobserver.png)

two honest reads. first, **77% of features clear icc 0.75 raw** (shape 100%, first-order 62%),
right in the published lidc inter-observer range (~60-85%), with shape beating first-order
exactly as expected. second, **real radiologists agree far more than the proxy implied**
(median icc 0.91 vs the proxy's 0.47): a uniform one-voxel erode/dilate is a deliberately
harsh stand-in, so it *under*-states real reproducibility. the proxy earns its keep because
the *qualitative* findings (shape > first-order, the floor helping, which features are robust)
replicate on the real readers, and the floor still lifts everything to ~0.99 (which survives
the degeneracy check below).

**texture is the exception worth dwelling on.** the 10 glcm features are by far the *most*
fragile under the proxy (median icc 0.36, lowest of any family, matching the literature's view
of texture as boundary-sensitive) yet look robust under the real readers (0.89). to test whether
that 0.89 was a real estimate or a small-cohort artifact, a **cohort-size sweep**
(`scripts/cohort_sweep.py`) grew the 4-rater set to **181 nodules** (a larger lidc download) and
recomputed the inter-observer icc at each size:

![two-panel sweep: top, median inter-observer ICC for shape, first-order and texture stays flat as the cohort grows from 20 to 181 nodules, with texture stable around 0.89 between shape at 0.93 and first-order at 0.81; bottom, the count of low-signal texture features stays flat near 4 to 5 out of 10 and never falls to zero](docs/figures/cohort_sweep.png)

the sweep settles it. texture reproducibility is **stable at ~0.89 from n=20 to n=181** (above
first-order's 0.81, below shape's 0.93), so the estimate is real, not underpowered. but **~4-5 of
the 10 glcm features stay flagged low-signal at every cohort size** (the bottom panel never falls
toward zero): roughly half the texture features are *genuinely near-constant* across lidc nodules,
so their iccs are ill-conditioned no matter how much data is added. the honest read is therefore
two-part: the signal-bearing texture features are reproducibly delineated, but the low-variance
half is a structural property of this 10-feature lite glcm on lidc, not a sample-size problem.
settling "stable vs underpowered" is exactly what the sweep is for, which is why the family median
is reported next to its low-signal count rather than alone. (under the aggressive proxy texture is
still the most fragile family, so the conservative reading stands.)

**caveats on the inter-observer result** (flagged by a radiomics review):

- **selection bias.** n=54 is the nodules *all four* radiologists drew, which are the larger,
  more conspicuous ones and inherently easier to delineate consistently. so this icc is an upper
  bound, inflated relative to the full nodule distribution; pairwise icc across whichever readers
  annotated each nodule, or a staple consensus, would be less biased.
- **the floored ~0.99 survives a degeneracy check.** if the -300 hu floor merely stripped the
  fuzzy margins where readers disagree and left the same dense core, the four masks would become
  identical and the floored icc would be tautological (identical masks -> identical features). so
  the pipeline reports the **mean pairwise dice across the four masks**: it is **0.785 raw -> 0.935
  floored**. the floor does tighten agreement (it drops some disagreed-upon low-hu margin), but the
  masks stay distinct (0.935, not ~1.0), so the near-perfect feature icc is genuine robustness, not
  identical inputs. (the raw inter-rater dice ~0.79 is itself right in the published lidc range.)
- **the proxy is conservative about magnitude, not every error mode.** a uniform erode/dilate
  over-states *net volume and boundary leakage*, but it does not reproduce the localized,
  shape-distorting topological mistakes real readers make.

**2. feature vs malignancy, with three caveats that matter.** nodule **size** is the strongest
predictor of the malignancy rating:

| feature | pearson r | auc | spearman w/ volume |
|---|---|---|---|
| shape_EquivalentDiameter | +0.72 | 0.94 | +1.00 (definitional) |
| shape_SurfaceArea | +0.59 | 0.95 | +0.98 (definitional) |
| shape_VoxelVolume | +0.55 | 0.94 | (size) |
| shape_Sphericity | -0.52 | 0.15 | -0.72 |

but that auc must be read against three caveats the pipeline and a radiomics review both flag:

- **the label is subjective, not pathology.** lidc malignancy is the radiologists' *suspicion*,
  and radiologists use size as a primary malignancy cue (fleischner / lung-rads). so size
  predicting the rating at auc 0.94 is a **confirmation of clinical triaging guidelines, not
  the discovery of an independent biomarker** (the size-only baseline on this label is a
  published auc 0.94-0.97, so this is exactly the expected number).
- **energy is a size proxy.** the volume-confound check flags `firstorder_Energy` at spearman
  0.81 with volume, so even where an intensity feature looks predictive it can be size in
  disguise, and is discounted.
- **nodules cluster within patients** (109 from 32), so the univariate stats are not
  independent; the run reports the patient count and a proper analysis needs cluster-robust or
  mixed-effects modeling (build plan).

**segmentation** dice was 0.47 on the threshold baseline (weak on real ct, as expected); the
features use the consensus mask, so this does not touch the feature results, and it is exactly
what the monai segmenter is for.

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
├── features.py         # radiomics-lite: shape + first-order + glcm texture (pyradiomics names)
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
scripts/   smoke_test.py · make_figures.py · cohort_sweep.py · download_data.md
configs/   default.yaml · lidc.yaml
tests/     test_core.py
```

## status

**verified end-to-end on both synthetic data and real lidc-idri.** the synthetic phantom
develops the method (floor recovery, volume-confound, the low-signal guard, all unit-tested,
exit-0); the real lidc run validates it (109 nodules / 32 patients, 4-radiologist
inter-observer reproducibility median icc 0.91 raw -> 0.99 floored, matching the published
range; a glcm texture family is in, and a cohort-size sweep to 181 nodules shows its
reproducibility is stable at ~0.89, not underpowered, with ~half the glcm features structurally
low-variance on lidc). next: train the monai segmenter to replace the dice-0.47 threshold
baseline, the remaining pyradiomics families on a texturally diverse cohort, and cluster-robust
stats (build plan above).
