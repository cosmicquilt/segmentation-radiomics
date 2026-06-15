"""seg_radiomics segmentation -> quantitative features -> outcome correlation

the clinical workflow cbig runs daily as a reproducible pipeline the numpy core
(segmentation metrics radiomics-lite feature extractor correlation qc and a
threshold-based baseline segmenter) runs with no heavy deps the learned segmenter
(monai/nnu-net) and canonical feature extractor (pyradiomics) plug in behind clear
interfaces
"""

__version__ = "0.1.0"
