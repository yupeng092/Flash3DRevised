"""Depth Anything V1 adapter for Flash3D's layered Gaussian decoder.

This mirrors :class:`DepthAnythingV2Extended` but binds the *first-generation*
Depth Anything model (https://github.com/LiheYoung/Depth-Anything).  The two key
differences from V2 are handled here:

1. The official V1 release is **relative depth** only.  Its forward output is
   per-image min-max normalised to ``[0, 1]`` (see ``relative`` below).  Flash3D
   recovers the absolute metric scale downstream through
   ``dataset.scale_pose_by_depth`` + ``misc.depth.estimate_depth_scale`` against
   the sparse COLMAP point cloud, so a relative prior is sufficient for
   single-image feed-forward training.

2. V1's ``DepthAnything`` lives in ``depth_anything/dpt.py`` (V2 uses
   ``depth_anything_v2/dpt.py``).  The HF space build of ``DepthAnything``
   inherits ``PyTorchModelHubMixin`` and takes a single ``config`` dict, so this
   adapter constructs the parent ``DPT_DINOv2`` directly with keyword args and
   loads the full checkpoint (which includes the DINOv2 backbone).

   ``dpt.py`` builds the DINOv2 backbone via
   ``torch.hub.load('torchhub/facebookresearch_dinov2_main', source='local')``.
   That path resolves under ``torch.hub.get_dir()``; this adapter points
   ``TORCH_HOME`` at the DA source dir so the vendored DINOv2 source (shipped
   inside the DA space repo under ``torchhub/``) is found without network.

Install the official Depth-Anything source under ``third_party`` and place a
matching full checkpoint (including the DINOv2 backbone) in ``weights`` as
configured in ``configs/model/depth/depth_anything_v1.yaml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.decoder.resnet_decoder import ResnetDepthDecoder, ResnetDecoder
from models.encoder.depth_anything_encoder import IntrinsicsHead
from models.encoder.resnet_encoder import ResnetEncoder


# V1 and V2 share the same DPT channel layout, but V1 is declared separately so
# the precompute script and this adapter can evolve independently.  ``readout``
# is passed at construction time and defaults to ``project`` (the V1 default).
DEPTH_ANYTHING_V1_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthAnythingV1Extended(nn.Module):
    """Drop-in Depth Anything V1 alternative to ``UniDepthExtended``."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        depth_cfg = cfg.model.depth
        encoder_name = depth_cfg.encoder
        if encoder_name not in DEPTH_ANYTHING_V1_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything V1 encoder: {encoder_name}")

        project_root = Path(__file__).resolve().parents[2]
        source_dir = project_root / Path(depth_cfg.source_dir)
        checkpoint = project_root / Path(depth_cfg.checkpoint)
        if not source_dir.joinpath("depth_anything", "dpt.py").is_file():
            raise FileNotFoundError(
                f"Depth Anything V1 source is missing at {source_dir}. "
                "Clone https://github.com/LiheYoung/Depth-Anything there."
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Depth Anything V1 checkpoint is missing: {checkpoint}"
            )
        sys.path.insert(0, str(source_dir))
        try:
            # The official V1 ``DepthAnything`` (HF space build) inherits
            # ``PyTorchModelHubMixin`` and takes a single ``config`` dict, which
            # is awkward to construct directly.  Its parent ``DPT_DINOv2`` takes
            # the plain keyword arguments we need, so we build that and then
            # load the full checkpoint (which includes the DINOv2 backbone).
            from depth_anything.dpt import DPT_DINOv2
        finally:
            sys.path.pop(0)

        # ``dpt.py`` calls ``torch.hub.load('torchhub/facebookresearch_dinov2_main',
        # source='local')`` to build the DINOv2 backbone structure.  With
        # ``source='local'`` that path is resolved relative to the *current
        # working directory*, not torch.hub.get_dir().  The vendored DINOv2
        # source lives at ``<source_dir>/torchhub/facebookresearch_dinov2_main/``,
        # so we temporarily chdir into ``source_dir`` for the construction and
        # restore the CWD afterwards (train.py later chdir's to the output dir).
        # ``readout`` maps to DPT_DINOv2's ``use_clstoken``.  The official V1
        # checkpoints were trained WITHOUT cls-token projection, so
        # ``use_clstoken=False`` is required for ``load_state_dict`` to match.
        # ``readout`` is retained for API symmetry with V2 but only "ignore"
        # (use_clstoken=False) matches the released weights.
        readout = getattr(depth_cfg, "readout", "ignore")
        use_clstoken = readout == "project"
        config = DEPTH_ANYTHING_V1_CONFIGS[encoder_name]
        previous_cwd = os.getcwd()
        os.chdir(str(source_dir))
        try:
            self.depth_model = DPT_DINOv2(
                encoder=encoder_name,
                features=config["features"],
                out_channels=config["out_channels"],
                use_clstoken=use_clstoken,
                localhub=True,
            )
        finally:
            os.chdir(previous_cwd)

        # DINOv2's MemEffAttention prefers xformers when importable, but the
        # installed xformers C++ extension only supports CUDA.  On CPU/NPU we
        # force the native-PyTorch fallback by clearing XFORMERS_AVAILABLE on
        # the attention module (now loaded by torch.hub under a dynamic name).
        for _name, _mod in list(sys.modules.items()):
            if _name.endswith("dinov2.layers.attention") and hasattr(_mod, "XFORMERS_AVAILABLE"):
                _mod.XFORMERS_AVAILABLE = False
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" in checkpoint_data:
            checkpoint_data = checkpoint_data["model"]
        info = self.depth_model.load_state_dict(checkpoint_data, strict=True)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"Depth Anything V1 checkpoint mismatch: {info}")
        self.depth_model.requires_grad_(not depth_cfg.freeze)

        # ``relative`` controls the official V1 per-image min-max normalisation.
        # It defaults to True (V1 is a relative-depth model).  Set False only if a
        # metric V1 fine-tune checkpoint is supplied.
        self.relative = getattr(depth_cfg, "relative", True)

        self.intrinsics_head = IntrinsicsHead(depth_cfg.initial_focal_ratio)
        self.encoder = ResnetEncoder(
            num_layers=cfg.model.backbone.num_layers,
            pretrained=cfg.model.backbone.weights_init == "pretrained",
            bn_order=cfg.model.backbone.resnet_bn_order,
        )
        if cfg.model.backbone.depth_cond:
            self.encoder.encoder.conv1 = nn.Conv2d(
                4,
                self.encoder.encoder.conv1.out_channels,
                kernel_size=self.encoder.encoder.conv1.kernel_size,
                padding=self.encoder.encoder.conv1.padding,
                stride=self.encoder.encoder.conv1.stride,
            )

        models = {}
        if cfg.model.gaussians_per_pixel > 1:
            models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
        for index in range(cfg.model.gaussians_per_pixel):
            models[f"gauss_decoder_{index}"] = ResnetDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
            if cfg.model.one_gauss_decoder:
                break
        self.models = nn.ModuleDict(models)

        self.parameters_to_train = [
            {"params": self.encoder.parameters()},
            {"params": self.models.parameters()},
        ]
        if not depth_cfg.freeze:
            self.parameters_to_train.append({"params": self.depth_model.parameters(), "lr": depth_cfg.finetune_lr})
        if depth_cfg.intrinsics_source == "learned":
            self.parameters_to_train.append({"params": self.intrinsics_head.parameters()})

    def get_parameter_groups(self):
        return self.parameters_to_train

    def _infer_depth(self, image: torch.Tensor) -> torch.Tensor:
        """ImageNet normalisation and aspect-ratio-preserving multiple-of-14 input.

        V1's ``forward`` may or may not min-max normalise internally depending on
        the checkpoint; applying the official per-image normalisation here is
        idempotent when the model already normalises and correct when it does not.
        """
        height, width = image.shape[-2:]
        long_side = int(self.cfg.model.depth.input_size)
        scale = long_side / max(height, width)
        resized_h = max(14, round(height * scale / 14) * 14)
        resized_w = max(14, round(width * scale / 14) * 14)
        resized = F.interpolate(image, (resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
        normalised = (resized - image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None])
        normalised = normalised / image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        context = torch.enable_grad() if not self.cfg.model.depth.freeze else torch.no_grad()
        with context:
            depth = self.depth_model(normalised)
        # Normalise to [B, 1, H, W] regardless of the backbone's return shape.
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, :1]
        if self.relative:
            batch = depth.shape[0]
            flat = depth.reshape(batch, -1)
            d_min = flat.amin(dim=1)
            d_max = flat.amax(dim=1)
            depth = (depth - d_min[:, None, None, None]) / ((d_max - d_min)[:, None, None, None] + 1e-6)
        depth = F.interpolate(depth, (height, width), mode="bilinear", align_corners=False, antialias=True)
        return depth.clamp_min(self.cfg.model.depth.min_depth)

    def _source_intrinsics(self, inputs: dict, image: torch.Tensor) -> torch.Tensor:
        use_provided = self.cfg.model.depth.intrinsics_source == "provided"
        if use_provided and ("K_src", 0) in inputs:
            return inputs[("K_src", 0)].to(dtype=image.dtype)
        return self.intrinsics_head(image)

    def forward(self, inputs):
        image = inputs["color_aug", 0, 0]
        # Re10K already supports the historical ``unidepth`` key for an
        # externally precomputed frozen depth prior.  Reuse that transport key
        # so NPU pre-training can omit the frozen ViT-B forward pass entirely.
        cached_depth = inputs.get(("unidepth", 0, 0))
        if cached_depth is None:
            depth = self._infer_depth(image)
        else:
            depth = cached_depth.to(device=image.device, dtype=image.dtype).clamp_min(
                self.cfg.model.depth.min_depth
            )
        source_intrinsics = self._source_intrinsics(inputs, image)
        outputs_gauss = {
            ("K_src", 0): source_intrinsics,
            ("inv_K_src", 0): torch.linalg.inv(source_intrinsics),
        }

        conditioned_image = torch.cat((image, depth / self.cfg.model.depth.depth_normalizer), dim=1) if self.cfg.model.backbone.depth_cond else image
        encoded_features = self.encoder(conditioned_image)
        if self.cfg.model.gaussians_per_pixel > 1:
            depth_offsets = self.models["depth"](encoded_features)
            depth_offsets[("depth", 0)] = rearrange(
                depth_offsets[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1
            )
            layered_depth = torch.cumsum(
                torch.cat((depth[:, None], depth_offsets[("depth", 0)]), dim=1), dim=1
            )
            outputs_gauss[("depth", 0)] = rearrange(
                layered_depth, "b n c h w -> (b n) c h w", n=self.cfg.model.gaussians_per_pixel
            )
        else:
            outputs_gauss[("depth", 0)] = depth

        gaussian_outputs = {}
        for index in range(self.cfg.model.gaussians_per_pixel):
            prediction = self.models[f"gauss_decoder_{index}"](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gaussian_outputs |= prediction
                break
            for key, value in prediction.items():
                gaussian_outputs[key] = value if index == 0 else torch.cat((gaussian_outputs[key], value), dim=1)
        for key, value in gaussian_outputs.items():
            gaussian_outputs[key] = rearrange(value, "b n ... -> (b n) ...")
        return outputs_gauss | gaussian_outputs
