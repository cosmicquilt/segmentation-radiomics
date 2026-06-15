# getting real lung-ct data

the pipeline runs immediately on **synthetic** volumes (no download). for real
results, lung ct is the path of least resistance (per the project plan). two good
options:

## option a: medical segmentation decathlon, task06_lung (easiest)

- site: <http://medicaldecathlon.com/>
- download `Task06_Lung.tar` (lung tumour segmentation). nifti format, already split
  into `imagesTr` / `labelsTr`, so no dicom wrangling.
- layout:
  ```
  data/Task06_Lung/
  ├── imagesTr/  lung_001.nii.gz ...
  ├── labelsTr/  lung_001.nii.gz ...
  └── dataset.json
  ```
- load with `nibabel`, this is the cleanest route to a trained monai/nnu-net model.

## option b: tcia, nsclc-radiomics or lidc-idri (closest to cbig's domain)

- site: <https://www.cancerimagingarchive.net/>
- **nsclc-radiomics**: ct + tumour segmentations + clinical outcomes (survival),
  ideal for the features -> outcome correlation step.
- **lidc-idri**: lung nodules with radiologist annotations + malignancy ratings,
  ideal for a features -> malignancy correlation.
- these are dicom/dicom-seg, use the nbia data retriever and read with `SimpleITK` or
  `pydicom`. heavier to wrangle than the decathlon, but on-mission.

## lidc-idri via pylidc (recommended, gives a REAL label)

the loader is already written (`src/seg_radiomics/data/lidc.py`), it pulls the
consensus nodule mask **and** the radiologist malignancy rating, so the
feature -> outcome correlation runs against a real clinical-ish label instead of a
manufactured one.

1. **get a subset of scans** (not the full 133 GB) via `tcia_utils` or the nbia data
   retriever. ~50-100 patients is plenty:
   ```python
   from tcia_utils import nbia
   nbia.downloadSeries(nbia.getSeries(collection="LIDC-IDRI")[:80], path="data/lidc")
   ```
2. **install pylidc + configure it** to find the dicoms:
   ```bash
   pip install pylidc pandas
   ```
   create `~/.pylidcrc` (windows: `C:\Users\<you>\pylidc.conf`):
   ```ini
   [dicom]
   path = /absolute/path/to/data/lidc/LIDC-IDRI
   warn = True
   ```
3. **run it:**
   ```bash
   python -m seg_radiomics.cli run --config configs/lidc.yaml
   ```
   `configs/lidc.yaml` sets `features.use_gt_mask: true`, so features come from the
   consensus mask (honest feature vs malignancy correlation), while dice still reports
   the threshold/monai segmenter vs that consensus.

## then upgrade the segmenter

swap the threshold baseline for a trained monai `UNet` (`segmentation/model.py`, dice
loss) and radiomics-lite for pyradiomics. the orchestration is unchanged.

> never commit clinical imaging data, `data/` is in `.gitignore`.
