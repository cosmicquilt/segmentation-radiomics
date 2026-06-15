"""learned segmentation via monai (the production upgrade over the baseline)

stubbed on purpose defines the interface and fails with a clear actionable message
if monai isnt installed so importing the package never forces a heavy dependency the
threshold baseline (baseline.threshold_segment) makes the whole pipeline run today
wire training in here to go from a dice floor to a real model

suggested path (per the spec) a 3d unet or nnu-net via monai on the msd lung task or
nsclc-radiomics dice loss report dice/iou on a held-out split
"""
from __future__ import annotations


def build_unet(spatial_dims: int = 3, in_channels: int = 1, out_channels: int = 2, **kwargs):
    """construct a monai unet requires pip install monai"""
    try:
        from monai.networks.nets import UNet
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MONAI is required for the learned segmenter. Install it with "
            "`pip install monai`. Until then, use segmentation.baseline.threshold_segment."
        ) from exc
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=kwargs.pop("channels", (16, 32, 64, 128, 256)),
        strides=kwargs.pop("strides", (2, 2, 2, 2)),
        num_res_units=kwargs.pop("num_res_units", 2),
        **kwargs,
    )


def train_segmenter(cfg: dict):
    """train the monai unet not yet implemented see the readme for the plan"""
    raise NotImplementedError(
        "Learned segmentation training is the next build step. The pipeline runs "
        "end-to-end today with the threshold baseline; swap it for build_unet() + a "
        "MONAI Dice-loss training loop here. See README 'Build plan'."
    )
