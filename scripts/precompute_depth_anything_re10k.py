#!/usr/bin/env python3
"""Precompute frozen Depth Anything depth maps in Flash3D's RE10K format.

Supports both Depth Anything V1 (official relative depth, the NPU pre-training
default) and Depth Anything V2 (metric).  Output files are
``OUTPUT/{train,test}/SEQUENCE/TIMESTAMP.png`` and carry the
``min_value``/``max_value`` PNG metadata consumed by ``datasets/re10k.py``.
Use them with ``dataset.preload_depths=true dataset.depth_path=OUTPUT`` to
remove the frozen depth encoder from the NPU/CUDA training step.

V1 notes: the official V1 model is relative depth, so each map is per-image
min-max normalised to ``[0, 1]`` before being saved.  Flash3D recovers the
absolute metric scale at training time through ``scale_pose_by_depth`` +
``estimate_depth_scale`` against the sparse COLMAP point cloud, so a relative
cache is sufficient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.re10k import load_seq_data


# Per-version defaults: source layout, checkpoint, and the model class/configs
# to import.  Configs are imported lazily from the matching encoder adapter so a
# missing third_party checkout for the *other* version never blocks this script.
VERSION_DEFAULTS = {
    "v1": {
        "source_dir": Path("third_party/Depth-Anything"),
        "checkpoint": Path("weights/depth-anything/depth_anything_vitb14.pth"),
        "import_path": "depth_anything.dpt",
        "class_name": "DepthAnything",
        "configs_module": "models.encoder.depth_anything_v1_encoder",
        "configs_attr": "DEPTH_ANYTHING_V1_CONFIGS",
        "relative": True,
    },
    "v2": {
        "source_dir": Path("third_party/Depth-Anything-V2"),
        "checkpoint": Path("weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth"),
        "import_path": "depth_anything_v2.dpt",
        "class_name": "DepthAnythingV2",
        "configs_module": "models.encoder.depth_anything_encoder",
        "configs_attr": "DEPTH_ANYTHING_CONFIGS",
        "relative": False,
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True, help="RealEstate10K root containing train/test metadata")
    parser.add_argument("--output", type=Path, required=True, help="Depth cache root")
    parser.add_argument("--version", choices=tuple(VERSION_DEFAULTS), default="v1", help="Depth Anything generation")
    parser.add_argument("--source-dir", type=Path, default=None, help="Override the third_party source dir for the chosen version")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override the checkpoint path for the chosen version")
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl"), default="vitb")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum total images; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "npu" or name == "auto":
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                return torch.device("npu:0")
        except ImportError:
            if name == "npu":
                raise
    if name == "cuda" or name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if name == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cpu")


def load_model(version: str, source_dir: Path, checkpoint: Path, encoder: str, device: torch.device) -> tuple[torch.nn.Module, bool]:
    spec = VERSION_DEFAULTS[version]
    module_file = source_dir.joinpath(*spec["import_path"].split(".")).with_suffix(".py")
    if not module_file.is_file():
        raise FileNotFoundError(f"Missing Depth Anything {version.upper()} source: {module_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Depth Anything {version.upper()} checkpoint: {checkpoint}")

    configs = getattr(__import__(spec["configs_module"], fromlist=[spec["configs_attr"]]), spec["configs_attr"])
    if encoder not in configs:
        raise ValueError(f"Encoder {encoder} is not available for Depth Anything {version.upper()}")

    import importlib
    sys.path.insert(0, str(source_dir))
    try:
        model_cls = getattr(importlib.import_module(spec["import_path"]), spec["class_name"])
    finally:
        sys.path.pop(0)

    if version == "v1":
        # V1's DepthAnything takes an extra ``readout`` argument.
        model = model_cls(encoder=encoder, readout="project", **configs[encoder])
    else:
        model = model_cls(encoder=encoder, **configs[encoder])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device).eval(), spec["relative"]


def infer(model: torch.nn.Module, image: torch.Tensor, input_size: int, relative: bool) -> torch.Tensor:
    height, width = image.shape[-2:]
    ratio = input_size / max(height, width)
    resized_height = max(14, round(height * ratio / 14) * 14)
    resized_width = max(14, round(width * ratio / 14) * 14)
    image = F.interpolate(image, (resized_height, resized_width), mode="bilinear", align_corners=False, antialias=True)
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    depth = model((image - mean) / std)
    # Normalise to [B, 1, H, W] regardless of the backbone's return shape.
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    elif depth.dim() == 4 and depth.shape[1] != 1:
        depth = depth[:, :1]
    if relative:
        # Official V1 relative depth: per-image min-max normalisation to [0, 1].
        batch = depth.shape[0]
        flat = depth.reshape(batch, -1)
        d_min = flat.amin(dim=1)
        d_max = flat.amax(dim=1)
        depth = (depth - d_min[:, None, None, None]) / ((d_max - d_min)[:, None, None, None] + 1e-6)
    depth = F.interpolate(depth, (height, width), mode="bilinear", align_corners=False, antialias=True).clamp_min(1e-4)
    return depth


def save_depth(depth: torch.Tensor, path: Path) -> None:
    value = depth.squeeze().detach().float().cpu().numpy()
    low, high = float(value.min()), float(value.max())
    span = max(high - low, 1e-6)
    encoded = np.round((value - low) / span * (2**16 - 1)).clip(0, 2**16 - 1).astype(np.uint16)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("min_value", repr(low))
    metadata.add_text("max_value", repr(high))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="I;16").save(path, pnginfo=metadata)


def main() -> None:
    args = arguments()
    spec = VERSION_DEFAULTS[args.version]
    source_dir = args.source_dir or (PROJECT_ROOT / spec["source_dir"])
    checkpoint = args.checkpoint or (PROJECT_ROOT / spec["checkpoint"])
    device = select_device(args.device)
    model, relative = load_model(args.version, source_dir, checkpoint, args.encoder, device)
    processed = 0
    for split in args.splits:
        sequences = load_seq_data(args.data_path, split)
        work = ((key, timestamp) for key, data in sequences.items() for timestamp in data["timestamps"])
        for sequence, timestamp in tqdm(work, desc=f"Depth Anything {args.version.upper()} {split}"):
            image_path = args.data_path / split / sequence / f"{timestamp}.jpg"
            output_path = args.output / split / sequence / f"{timestamp}.png"
            if not image_path.is_file():
                continue
            if output_path.is_file() and not args.overwrite:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
                tensor = to_tensor(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                depth = infer(model, tensor, args.input_size, relative)
            save_depth(depth, output_path)
            processed += 1
            if args.limit and processed >= args.limit:
                print(f"Stopped after {processed} images as requested.")
                return
    print(f"Saved {processed} Depth Anything {args.version.upper()} maps under {args.output.resolve()}")


if __name__ == "__main__":
    main()
