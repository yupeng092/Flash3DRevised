#!/usr/bin/env python3
"""GSFixer-style CPU post-processor for Flash3D Gaussian point clouds.

It refines the ``gaussians.pt`` produced by ``benchmark_cpu.py`` (or any file in
the same format consumed by ``render_cpu_alpha.py``) *before* rendering, so the
improvement is multi-view consistent rather than a per-frame 2D patch.

No learned weights, no CUDA, no differentiable rasterizer.  Every grid-mode
stage is a deterministic tensor op that exploits Flash3D's per-pixel layout:

    flat_index = layer * (H * W) + pixel,   pixel = y * W + x   (row-major)

The grid stages carry a ``(layers, H, W)`` valid mask and never delete points
until the grid work is done, so floater removal never breaks the neighbourhood
structure that smoothing and hole-fill rely on.

Stages (pick via --stages):
  1. prune-padding   drop Gaussians from Flash3D's zero-padded image border
  2. prune-floaters  drop depth outliers (the classic "floating blobs")
  3. smooth          edge-preserving bilateral filter on SH/scale/opacity
  4. hole-fill       re-seed empty source-view pixels from neighbours
  5. densify         split the largest Gaussians into children for finer detail

Output is a drop-in replacement for the input file: the same
``{"gaussians": {...}, "metadata": {...}}`` structure with all original keys
preserved, plus a ``postprocess`` record in the metadata.

Usage:
  python postprocess_gaussians.py \
      --input outputs/courtyard_benchmark/gaussians.pt \
      --output outputs/courtyard_benchmark/gaussians_refined.pt

  python render_cpu_alpha.py \
      --gaussians outputs/courtyard_benchmark/gaussians_refined.pt \
      --poses outputs/courtyard_benchmark/render_poses.pt \
      --output outputs/courtyard_render_refined ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# Loading / grid reconstruction
# --------------------------------------------------------------------------- #
def load_payload(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload["gaussians"] if "gaussians" in payload else payload
    metadata = payload.get("metadata", {}) or {}
    required = {"xyz", "opacity", "scaling", "rotation", "features_dc"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"Missing Gaussian fields: {sorted(missing)}")
    gaussians = {k: v.detach().float().cpu() for k, v in raw.items()}
    gaussians["opacity"] = gaussians["opacity"].reshape(-1, 1)
    gaussians["features_dc"] = gaussians["features_dc"].reshape(-1, 3)
    gaussians["scaling"] = gaussians["scaling"].reshape(-1, 3)
    gaussians["rotation"] = gaussians["rotation"].reshape(-1, 4)
    if "features_rest" in gaussians:
        n = gaussians["xyz"].shape[0]
        gaussians["features_rest"] = gaussians["features_rest"].reshape(n, -1)
    return gaussians, metadata


@dataclass
class Grid:
    layers: int
    height: int
    width: int
    input_h: int
    input_w: int
    pad_y: int
    pad_x: int

    @property
    def hw(self) -> int:
        return self.height * self.width

    @property
    def inner_mask(self) -> torch.Tensor:
        spatial = torch.zeros((self.height, self.width), dtype=torch.bool)
        spatial[self.pad_y:self.pad_y + self.input_h,
                self.pad_x:self.pad_x + self.input_w] = True
        return spatial.unsqueeze(0).repeat(self.layers, 1, 1)


def reconstruct_grid(metadata: dict, n_points: int) -> Grid | None:
    layers = int(metadata.get("gaussians_per_pixel", 1) or 1)
    input_h, input_w = metadata.get("input_size_hw", (0, 0))
    padded_h, padded_w = metadata.get("padded_size_hw", (input_h, input_w))
    if not (input_h and input_w and padded_h and padded_w):
        return None
    if n_points != layers * padded_h * padded_w:
        return None
    return Grid(layers, padded_h, padded_w, input_h, input_w,
                (padded_h - input_h) // 2, (padded_w - input_w) // 2)


def to_image(attr: torch.Tensor, grid: Grid) -> torch.Tensor:
    """(N, C) flat, layer-major -> (layers, C, H, W). N must equal layers*hw."""
    n, c = attr.shape
    return attr.reshape(grid.layers, grid.hw, c).permute(0, 2, 1).reshape(
        grid.layers, c, grid.height, grid.width
    )


def from_image(img: torch.Tensor) -> torch.Tensor:
    """(layers, C, H, W) -> (N, C) flat, layer-major."""
    layers, c, h, w = img.shape
    return img.permute(0, 2, 3, 1).reshape(-1, c).contiguous()


def apply_mask(g: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    mask = mask.reshape(-1)
    return {k: v[mask] for k, v in g.items()}


# --------------------------------------------------------------------------- #
# Stage 1: slice off the zero-padded border (junk from zero-padding)
# --------------------------------------------------------------------------- #
def slice_to_inner(g: dict[str, torch.Tensor], grid: Grid) -> tuple[dict, Grid]:
    """Crop the padded cloud to the unpadded image region.

    Returns the cropped cloud (flat, length = layers*input_h*input_w) and the
    inner grid that describes it.  Padding is removed outright because those
    Gaussians come from a zero-padded image and are never worth re-seeding.
    """
    inner = Grid(grid.layers, grid.input_h, grid.input_w,
                 grid.input_h, grid.input_w, 0, 0)
    out = {}
    for key, val in g.items():
        img = to_image(val, grid)
        cropped = img[:, :, grid.pad_y:grid.pad_y + grid.input_h,
                            grid.pad_x:grid.pad_x + grid.input_w]
        out[key] = from_image(cropped)
    return out, inner


# --------------------------------------------------------------------------- #
# Stage 2: floater / depth-outlier mask (per layer, 3x3 median + MAD)
# --------------------------------------------------------------------------- #
def floater_valid_image(g: dict[str, torch.Tensor], grid: Grid,
                        k_mad: float, min_opacity: float) -> torch.Tensor:
    """Return a (layers, H, W) bool mask, True for Gaussians to keep."""
    z = g["xyz"][:, 2]
    finite = torch.isfinite(z) & torch.isfinite(g["scaling"]).all(dim=-1)
    valid_depth = finite & (z > 1e-4)
    keep = valid_depth & (g["opacity"].reshape(-1) >= min_opacity)

    z_img = to_image(z.reshape(-1, 1), grid)            # (L, 1, H, W)
    unfold = F.unfold(z_img, kernel_size=3, padding=1)  # (L, 9, H*W)
    med = torch.median(unfold, dim=1).values            # (L, H*W)
    dev = (unfold - med.unsqueeze(1)).abs()
    mad = torch.median(dev, dim=1).values               # (L, H*W)
    center = z_img.reshape(grid.layers, grid.hw)
    outlier = (center - med).abs() > k_mad * (mad + 1e-6)
    keep_img = (~outlier).reshape(grid.layers, grid.height, grid.width)
    keep_img = keep_img & keep.reshape(grid.layers, grid.height, grid.width)
    return keep_img


# --------------------------------------------------------------------------- #
# Stage 3: edge-preserving bilateral smoothing (per layer, image space)
# --------------------------------------------------------------------------- #
def _bilateral(attr_img: torch.Tensor, guide_img: torch.Tensor,
               valid_img: torch.Tensor, spatial_sigma: float,
               range_sigma: float, strength: float, kernel: int) -> torch.Tensor:
    """Bilateral filter a (layers, C, H, W) attribute guided by (layers, 3, H, W) RGB.

    Neighbours flagged invalid in ``valid_img`` (layers, H, W) are excluded from
    the weighted average so removed floaters do not pollute their neighbours.
    """
    layers, c, h, w = attr_img.shape
    pad = kernel // 2
    neigh = F.unfold(attr_img, kernel_size=kernel, padding=pad)
    neigh = neigh.reshape(layers, c, kernel * kernel, h * w)
    center = attr_img.reshape(layers, c, 1, h * w)

    guide = F.unfold(guide_img, kernel_size=kernel, padding=pad)
    guide = guide.reshape(layers, 3, kernel * kernel, h * w)
    g_center = guide_img.reshape(layers, 3, 1, h * w)
    diff = guide - g_center
    range_w = torch.exp(-(diff * diff).sum(dim=1) / (2.0 * range_sigma * range_sigma))

    coords = torch.arange(-(kernel // 2), kernel // 2 + 1, dtype=torch.float32)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_w = torch.exp(-(gx * gx + gy * gy) / (2.0 * spatial_sigma * spatial_sigma))
    spatial_w = spatial_w.reshape(1, kernel * kernel, 1)

    valid_neigh = F.unfold(valid_img.float().unsqueeze(1), kernel_size=kernel,
                           padding=pad).reshape(layers, 1, kernel * kernel, h * w)
    weight = (spatial_w * range_w).unsqueeze(1) * valid_neigh       # (L, 1, k*k, H*W)
    wsum = weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
    filtered = (neigh * weight).sum(dim=2, keepdim=True) / wsum      # (L, C, 1, H*W)
    filtered = filtered.reshape(layers, c, h, w)
    return (1.0 - strength) * attr_img + strength * filtered


def smooth_attributes(g: dict[str, torch.Tensor], grid: Grid, valid: torch.Tensor,
                      spatial_sigma: float, range_sigma: float,
                      strength: float, kernel: int) -> dict[str, torch.Tensor]:
    color = (0.5 + SH_C0 * g["features_dc"]).clamp(0.0, 1.0)
    guide = to_image(color, grid)
    v_img = valid.reshape(grid.layers, grid.height, grid.width)

    dc_img = _bilateral(to_image(g["features_dc"], grid), guide, v_img,
                        spatial_sigma, range_sigma, strength, kernel)
    g["features_dc"] = from_image(dc_img)

    log_s = torch.log(g["scaling"].clamp_min(1e-7))
    log_s_img = _bilateral(to_image(log_s, grid), guide, v_img,
                           spatial_sigma, range_sigma, strength, kernel)
    g["scaling"] = torch.exp(from_image(log_s_img))

    op_img = _bilateral(to_image(g["opacity"], grid), guide, v_img,
                        spatial_sigma, range_sigma, strength, kernel)
    g["opacity"] = from_image(op_img).clamp(0.0, 1.0)

    if "features_rest" in g:
        rest_img = _bilateral(to_image(g["features_rest"], grid), guide, v_img,
                              spatial_sigma, range_sigma, strength, kernel)
        g["features_rest"] = from_image(rest_img)
    # Rotation is left untouched: quaternion averaging is ill-conditioned.
    return g


# --------------------------------------------------------------------------- #
# Stage 4: hole fill — re-seed empty source-view pixels from neighbours
# --------------------------------------------------------------------------- #
def hole_fill(g: dict[str, torch.Tensor], grid: Grid, valid: torch.Tensor,
              passes: int) -> tuple[dict, torch.Tensor]:
    """Dilate the valid mask so empty inner pixels inherit a neighbour's Gaussian."""
    attrs = {k: to_image(v, grid) for k, v in g.items()}
    filled = {k: v.clone() for k, v in attrs.items()}
    cur = valid.clone()
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for _ in range(passes):
        need = (~cur)
        if not need.any():
            break
        new_cur = cur.clone()
        for dy, dx in shifts:
            rolled_mask = torch.roll(cur, shifts=(dy, dx), dims=(1, 2))
            target = need & rolled_mask
            if not target.any():
                continue
            for key in attrs:
                rolled = torch.roll(filled[key], shifts=(dy, dx), dims=(2, 3))
                filled[key] = torch.where(target.unsqueeze(1).expand_as(rolled), rolled, filled[key])
            new_cur = new_cur | target
        cur = new_cur

    out = {k: from_image(v) for k, v in filled.items()}
    return out, cur


# --------------------------------------------------------------------------- #
# Stage 5: densify by splitting the largest Gaussians (3-D, post-flatten)
# --------------------------------------------------------------------------- #
def densify(g: dict[str, torch.Tensor], frac: float, split_count: int,
            scale_factor: float, max_growth: float) -> dict[str, torch.Tensor]:
    """Split the top-``frac`` largest Gaussians into ``split_count`` children.

    Each child keeps the parent's rotation and SH coefficients; its scale along
    the largest axis is reduced by ``scale_factor`` and its position is offset
    along that axis so the children tile the parent's footprint.  Opacity is
    redistributed so the aggregate alpha is preserved.
    """
    if split_count < 2 or frac <= 0:
        return g
    scaling = g["scaling"]
    s_max = scaling.max(dim=-1).values
    n = s_max.numel()
    n_split = min(int(n * frac), int(n * max_growth))
    if n_split <= 0:
        return g
    threshold = torch.quantile(s_max, 1.0 - frac)
    split_idx = torch.where(s_max >= threshold)[0]
    if split_idx.numel() == 0:
        return g
    if split_idx.numel() > n_split:
        split_idx = split_idx[torch.topk(s_max[split_idx], n_split).indices]

    axis = scaling[split_idx].argmax(dim=-1)                 # (M,)
    xyz = g["xyz"][split_idx]                                # (M, 3)
    op = g["opacity"][split_idx]                             # (M, 1)
    sc = scaling[split_idx]                                  # (M, 3)
    half = s_max[split_idx] * 0.5                            # (M,)

    locs = torch.linspace(-1.0, 1.0, split_count, dtype=half.dtype).unsqueeze(1) * half  # (split_count, M)
    offsets = []
    for i in range(split_count):
        off = torch.zeros_like(xyz)
        off[:, axis] = locs[i]
        offsets.append(off)

    child_opacity = 1.0 - (1.0 - op).pow(1.0 / split_count)  # preserve aggregate alpha
    child_scale = sc.clone()
    for a in range(3):
        child_scale[:, a] = torch.where(axis == a, child_scale[:, a] / scale_factor, child_scale[:, a])

    new_xyz = torch.cat([xyz + off for off in offsets], dim=0)
    new_op = child_opacity.repeat(split_count, 1)
    new_sc = child_scale.repeat(split_count, 1)
    new_rot = g["rotation"][split_idx].repeat(split_count, 1)
    new_dc = g["features_dc"][split_idx].repeat(split_count, 1)
    rest_key = "features_rest" in g
    new_rest = g["features_rest"][split_idx].repeat(split_count, 1) if rest_key else None

    keep_mask = torch.ones(n, dtype=torch.bool)
    keep_mask[split_idx] = False
    out = {
        "xyz": torch.cat([g["xyz"][keep_mask], new_xyz], dim=0),
        "opacity": torch.cat([g["opacity"][keep_mask], new_op], dim=0),
        "scaling": torch.cat([g["scaling"][keep_mask], new_sc], dim=0),
        "rotation": torch.cat([g["rotation"][keep_mask], new_rot], dim=0),
        "features_dc": torch.cat([g["features_dc"][keep_mask], new_dc], dim=0),
    }
    if rest_key:
        out["features_rest"] = torch.cat([g["features_rest"][keep_mask], new_rest], dim=0)
    return out


# --------------------------------------------------------------------------- #
# 3-D fallback for non-grid clouds (e.g. colmap_to_gaussians.py output)
# --------------------------------------------------------------------------- #
def floater_mask_3d(g: dict[str, torch.Tensor], k_factor: float,
                    min_opacity: float) -> torch.Tensor:
    """Drop isolated points whose k-th nearest-neighbour distance is an outlier."""
    xyz = g["xyz"]
    n = xyz.shape[0]
    keep = torch.isfinite(xyz).all(dim=-1) & (g["opacity"].reshape(-1) >= min_opacity)
    if n < 64:
        return keep
    m = min(4096, n)
    anchor = torch.randperm(n)[:m]
    d = torch.cdist(xyz, xyz[anchor])                       # (n, m)
    kth = min(4, m - 1)
    knn = d.sort(dim=1).values[:, kth]
    thresh = torch.median(knn) * k_factor
    return keep & (knn <= thresh)


def smooth_attributes_3d(g: dict[str, torch.Tensor], spatial_sigma: float,
                         range_sigma: float, strength: float) -> dict[str, torch.Tensor]:
    """Lightweight 3-D bilateral via anchor subsampling, chunked over points."""
    xyz = g["xyz"]
    n = xyz.shape[0]
    if n < 64:
        return g
    color = (0.5 + SH_C0 * g["features_dc"]).clamp(0.0, 1.0)
    m = min(2048, n)
    anchor = torch.randperm(n)[:m]
    anc_xyz = xyz[anchor]
    anc_col = color[anchor]

    def filt(attr: torch.Tensor) -> torch.Tensor:
        out = attr.clone()
        chunk = 8192
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = torch.cdist(xyz[s:e], anc_xyz)
            w_s = torch.exp(-d * d / (2.0 * spatial_sigma * spatial_sigma))
            cd = torch.cdist(color[s:e], anc_col)
            w_r = torch.exp(-cd * cd / (2.0 * range_sigma * range_sigma))
            w = w_s * w_r
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            smoothed = (w.unsqueeze(1) * attr[anchor].unsqueeze(0)).sum(dim=1)
            out[s:e] = (1.0 - strength) * attr[s:e] + strength * smoothed
        return out

    g["features_dc"] = filt(g["features_dc"])
    g["scaling"] = torch.exp(filt(torch.log(g["scaling"].clamp_min(1e-7))))
    g["opacity"] = filt(g["opacity"]).clamp(0.0, 1.0)
    if "features_rest" in g:
        g["features_rest"] = filt(g["features_rest"])
    return g


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def refine(gaussians: dict[str, torch.Tensor], metadata: dict,
           stages: Iterable[str], *, floater_k: float, min_opacity: float,
           smooth_strength: float, spatial_sigma: float, range_sigma: float,
           smooth_kernel: int, densify_frac: float, split_count: int,
           scale_factor: float, max_growth: float, hole_passes: int) -> tuple[dict, dict]:
    stages = list(stages)
    n0 = gaussians["xyz"].shape[0]
    grid = reconstruct_grid(metadata, n0)
    stats = {"input_points": n0, "grid_mode": grid is not None}

    valid: torch.Tensor | None = None  # (layers, H, W) carry mask in grid mode

    if grid is not None:
        if "prune-padding" in stages:
            gaussians, grid = slice_to_inner(gaussians, grid)
            stats["after_prune_padding"] = gaussians["xyz"].shape[0]
        valid = torch.ones(grid.layers, grid.height, grid.width, dtype=torch.bool)

        if "prune-floaters" in stages:
            valid = valid & floater_valid_image(gaussians, grid, floater_k, min_opacity)
            stats["after_prune_floaters"] = int(valid.sum().item())

        if "smooth" in stages and smooth_strength > 0:
            gaussians = smooth_attributes(gaussians, grid, valid, spatial_sigma,
                                          range_sigma, smooth_strength, smooth_kernel)
            stats["after_smooth"] = int(valid.sum().item())

        if "hole-fill" in stages:
            before = int(valid.sum().item())
            gaussians, valid = hole_fill(gaussians, grid, valid, hole_passes)
            stats["after_hole_fill"] = int(valid.sum().item())

        gaussians = apply_mask(gaussians, valid.reshape(-1))
    else:
        # 3-D fallback for clouds without a recoverable pixel grid.
        if "prune-floaters" in stages:
            gaussians = apply_mask(gaussians, floater_mask_3d(gaussians, floater_k * 2.0, min_opacity))
            stats["after_prune_floaters"] = gaussians["xyz"].shape[0]
        if "smooth" in stages and smooth_strength > 0:
            gaussians = smooth_attributes_3d(gaussians, spatial_sigma, range_sigma, smooth_strength)
            stats["after_smooth"] = gaussians["xyz"].shape[0]

    if "densify" in stages and densify_frac > 0:
        gaussians = densify(gaussians, densify_frac, split_count, scale_factor, max_growth)
        stats["after_densify"] = gaussians["xyz"].shape[0]

    stats["output_points"] = gaussians["xyz"].shape[0]
    return gaussians, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
ALL_STAGES = ["prune-padding", "prune-floaters", "smooth", "hole-fill", "densify"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    p.add_argument("--floater-k", type=float, default=4.0,
                   help="Depth-outlier threshold in MAD units (larger = less aggressive).")
    p.add_argument("--min-opacity", type=float, default=0.01)
    p.add_argument("--smooth-strength", type=float, default=0.5,
                   help="0 keeps attributes, 1 replaces them with the filtered result.")
    p.add_argument("--spatial-sigma", type=float, default=1.2,
                   help="Bilateral spatial sigma in pixels (grid mode).")
    p.add_argument("--range-sigma", type=float, default=0.1,
                   help="Bilateral range sigma in RGB units (0-1).")
    p.add_argument("--smooth-kernel", type=int, default=5, choices=[3, 5, 7])
    p.add_argument("--densify-frac", type=float, default=0.10,
                   help="Fraction of the largest Gaussians to split.")
    p.add_argument("--split-count", type=int, default=2,
                   help="Children per split Gaussian.")
    p.add_argument("--scale-factor", type=float, default=2.0,
                   help="Divide the split axis scale by this (2 => half-size children).")
    p.add_argument("--max-growth", type=float, default=0.5,
                   help="Cap on added points as a fraction of the current cloud.")
    p.add_argument("--hole-passes", type=int, default=2)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input.resolve()} ...")
    gaussians, metadata = load_payload(args.input)
    print(f"  points: {gaussians['xyz'].shape[0]}, stages: {args.stages}")

    t0 = time.perf_counter()
    refined, stats = refine(
        gaussians, metadata, args.stages,
        floater_k=args.floater_k, min_opacity=args.min_opacity,
        smooth_strength=args.smooth_strength, spatial_sigma=args.spatial_sigma,
        range_sigma=args.range_sigma, smooth_kernel=args.smooth_kernel,
        densify_frac=args.densify_frac, split_count=args.split_count,
        scale_factor=args.scale_factor, max_growth=args.max_growth,
        hole_passes=args.hole_passes,
    )
    elapsed = time.perf_counter() - t0

    out_payload = {
        "gaussians": {k: v.contiguous() for k, v in refined.items()},
        "metadata": {
            **metadata,
            "postprocess": {
                "stages": args.stages,
                "elapsed_seconds": elapsed,
                **stats,
            },
        },
    }
    torch.save(out_payload, args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "elapsed_seconds": round(elapsed, 3), **stats}, indent=2))


if __name__ == "__main__":
    main()
