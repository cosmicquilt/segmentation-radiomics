"""segmentation a numpy threshold baseline (runnable now) and a monai unet stub

the threshold segmenter is to project 2 what the zero-filled baseline is to project
1 a parameter-free floor that makes the whole pipeline run end-to-end before any
training the learned segmenter (model.py) is where monai/nnu-net plugs in
"""

from .baseline import threshold_segment

__all__ = ["threshold_segment"]
