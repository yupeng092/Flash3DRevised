#!/usr/bin/env python3
"""RealESRGAN super-resolution for rendered images.

Loads a RealESRGAN x4 model via spandrel, processes images in tiles with
overlap to avoid seams, and outputs 4x upscaled images.

Usage:
    python realesrgan_upscale.py \
        --input outputs/courtyard_colmap_midres/rgb \
        --output outputs/courtyard_colmap_sr/rgb \
        --model weights/realesrgan/RealESRGAN_x4.pth \
        --tile-size 256 --overlap 32 --threads 28
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from spandrel import ModelLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image directory or file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", type=Path, default=Path("weights/realesrgan/RealESRGAN_x4.pth"))
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for processing")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between tiles")
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--scale", type=int, default=4, help="Output is tile_size * scale")
    return parser.parse_args()


def process_tile(model: torch.nn.Module, tile: torch.Tensor) -> torch.Tensor:
    """Process a single tile: [1,3,H,W] -> [1,3,H*scale,W*scale]."""
    with torch.no_grad():
        return model(tile)


def blend_tiles(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    out_h: int,
    out_w: int,
    scale: int,
) -> np.ndarray:
    """Blend overlapping tiles with linear fade at borders."""
    result = np.zeros((3, out_h, out_w), dtype=np.float32)
    weight = np.zeros((3, out_h, out_w), dtype=np.float32)

    # Create a smooth weight mask (linear ramp in overlap region)
    mask = np.ones((3, tile_h * scale, tile_w * scale), dtype=np.float32)

    # Apply linear fade at borders
    fade = overlap * scale
    if fade > 0:
        for i in range(fade):
            w = (i + 1) / fade
            mask[:, :, i] *= w
            mask[:, :, -(i + 1)] *= w
            mask[:, i, :] *= w
            mask[:, -(i + 1), :] *= w

    for tile_np, (y, x) in zip(tiles, positions):
        sy = y * scale
        sx = x * scale
        th, tw = tile_np.shape[1], tile_np.shape[2]
        # Clip to output bounds
        eh = min(sy + th, out_h)
        ew = min(sx + tw, out_w)
        result[:, sy:eh, sx:ew] += tile_np[:, : eh - sy, : ew - sx] * mask[:, : eh - sy, : ew - sx]
        weight[:, sy:eh, sx:ew] += mask[:, : eh - sy, : ew - sx]

    result /= np.maximum(weight, 1e-8)
    return result


def upscale_image(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    scale: int,
    device: torch.device,
) -> np.ndarray:
    """Upscale a single image using tiled processing."""
    _, h, w = image.shape  # [3, H, W]
    out_h = h * scale
    out_w = w * scale

    # Calculate tile positions
    stride = tile_size - overlap
    y_positions = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not y_positions or y_positions[-1] + tile_size < h:
        y_positions.append(max(h - tile_size, 0))
    x_positions = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not x_positions or x_positions[-1] + tile_size < w:
        x_positions.append(max(w - tile_size, 0))

    # Remove duplicates and sort
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    total_tiles = len(y_positions) * len(x_positions)
    print(f"  Image {w}x{h} -> {out_w}x{out_h}, {total_tiles} tiles "
          f"({len(y_positions)}x{len(x_positions)})")

    tiles_out = []
    positions = []
    processed = 0

    for y in y_positions:
        for x in x_positions:
            # Extract tile
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)

            tile = image[:, y : y + th, x : x + tw]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device).float()

            # Pad if tile is smaller than tile_size
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile_tensor = F.pad(tile_tensor, (0, pad_w, 0, pad_h), mode="reflect")

            # Process
            out_tile = process_tile(model, tile_tensor)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                out_tile = out_tile[:, :, : th * scale, : tw * scale]

            tile_np = out_tile.squeeze(0).cpu().numpy()
            tiles_out.append(tile_np)
            positions.append((y, x))

            processed += 1
            if processed % 4 == 0:
                print(f"    {processed}/{total_tiles} tiles done")

    # Blend tiles
    result = blend_tiles(tiles_out, positions, tile_size, tile_size,
                         overlap, out_h, out_w, scale)
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    print(f"Loading RealESRGAN model from {args.model}...")
    descriptor = ModelLoader().load_from_file(str(args.model))
    model = descriptor.model.to(device).eval()
    print(f"  Model: {type(model).__name__}, scale={descriptor.scale}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Collect input files
    if args.input.is_file():
        input_files = [args.input]
    else:
        input_files = sorted(
            f for f in args.input.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        )

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(input_files)} images...\n")

    total_time = 0
    for i, in_path in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] {in_path.name}")
        image = np.array(Image.open(in_path).convert("RGB")).transpose(2, 0, 1) / 255.0

        t0 = time.perf_counter()
        result = upscale_image(model, image, args.tile_size, args.overlap,
                               args.scale, device)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Save
        out_array = (result * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        out_image = Image.fromarray(out_array.transpose(1, 2, 0), mode="RGB")
        out_path = args.output / in_path.name
        out_image.save(out_path)
        print(f"  Saved {out_path.name}: {out_image.size[0]}x{out_image.size[1]} "
              f"({out_path.stat().st_size // 1024}KB), {elapsed:.1f}s\n")

    print(f"Total time: {total_time:.1f}s, avg: {total_time/len(input_files):.1f}s/image")


if __name__ == "__main__":
    main()
