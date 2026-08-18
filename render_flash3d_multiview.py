#!/usr/bin/env python3
"""Render Flash3D point cloud using real COLMAP multi-view camera poses.

Transforms COLMAP world-to-camera poses into Flash3D's camera-0 coordinate
system, applies scale alignment, then renders with render_cpu_alpha.py's
3DGS rasterizer.

Usage:
    python render_flash3d_multiview.py \
        --gaussians outputs/courtyard_benchmark/gaussians.pt \
        --colmap-dir "D:/Python Project/courtyard/dslr_calibration_undistorted" \
        --output outputs/courtyard_flash3d_multiview \
        --ref-image "DSC_0286" \
        --render-height 256 --render-width 384
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SH_C0 = 0.28209479177387814


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gaussians", type=Path, required=True)
    p.add_argument("--colmap-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ref-image", default="DSC_0286", help="Reference image name (camera 0)")
    p.add_argument("--render-height", type=int, default=256)
    p.add_argument("--render-width", type=int, default=384)
    p.add_argument("--pose-indices", type=int, nargs="*", default=None,
                   help="Which COLMAP image indices to render (0-based). Default: 10 evenly spaced")
    p.add_argument("--scale", type=float, default=None,
                   help="Manual scale factor. If None, auto-estimate from depth medians")
    p.add_argument("--keep-ratio", type=float, default=0.35)
    return p.parse_args()


def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1 - 2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1 - 2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def read_colmap_images(path):
    images = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        images.append({
            "id": int(parts[0]),
            "qvec": [float(x) for x in parts[1:5]],
            "tvec": [float(x) for x in parts[5:8]],
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
        i += 1  # skip POINTS2D
    return images


def read_colmap_cameras(path):
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        cameras[cam_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(x) for x in parts[4:]],
        }
    return cameras


def read_colmap_points(path):
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float64)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load Flash3D gaussians
    print("Loading Flash3D gaussians...")
    data = torch.load(args.gaussians, map_location="cpu", weights_only=True)
    g = data["gaussians"]
    xyz_flash3d = g["xyz"].numpy()
    opacity = g["opacity"].numpy().reshape(-1)
    scaling = g["scaling"].numpy()
    rotation = g["rotation"].numpy()
    features_dc = g["features_dc"].numpy()

    # Filter by opacity
    valid = opacity >= 0.01
    xyz_flash3d = xyz_flash3d[valid]
    opacity = opacity[valid]
    scaling = scaling[valid]
    rotation = rotation[valid]
    features_dc = features_dc[valid]

    # Keep ratio
    if args.keep_ratio < 1.0:
        n_keep = max(1, int(len(xyz_flash3d) * args.keep_ratio))
        keep_idx = np.argsort(opacity)[-n_keep:]
        xyz_flash3d = xyz_flash3d[keep_idx]
        opacity = opacity[keep_idx]
        scaling = scaling[keep_idx]
        rotation = rotation[keep_idx]
        features_dc = features_dc[keep_idx]

    print(f"  {len(xyz_flash3d)} gaussians after filtering")

    # 2. Read COLMAP
    print("Reading COLMAP...")
    images = read_colmap_images(args.colmap_dir / "images.txt")
    cameras = read_colmap_cameras(args.colmap_dir / "cameras.txt")
    colmap_pts = read_colmap_points(args.colmap_dir / "points3D.txt")

    # Find reference image (camera 0)
    ref_img = None
    for img in images:
        if args.ref_image in img["name"]:
            ref_img = img
            break
    if ref_img is None:
        raise ValueError(f"Reference image {args.ref_image} not found in COLMAP")
    print(f"  Reference: {ref_img['name']}")

    R0 = quat_to_rot(ref_img["qvec"])
    t0 = np.array(ref_img["tvec"])

    # 3. Estimate scale
    if args.scale is not None:
        scale = args.scale
    else:
        pts_cam0 = (R0 @ colmap_pts.T).T + t0
        colmap_depths = pts_cam0[:, 2]
        colmap_depths = colmap_depths[colmap_depths > 0]
        colmap_depth_median = np.median(colmap_depths)
        flash3d_depth_median = np.median(xyz_flash3d[:, 2])
        scale = colmap_depth_median / flash3d_depth_median
    print(f"  Scale factor: {scale:.4f}")

    # Scale Flash3D point cloud
    xyz_scaled = xyz_flash3d * scale

    # 4. Compute relative poses (camera 0 -> camera i)
    print("\nComputing relative poses...")
    relative_poses = []
    for img in images:
        Ri = quat_to_rot(img["qvec"])
        ti = np.array(img["tvec"])
        # P_cami = Ri @ R0^T @ P_cam0 + (ti - Ri @ R0^T @ t0)
        R_rel = Ri @ R0.T
        t_rel = ti - R_rel @ t0
        trans_dist = np.linalg.norm(t_rel)
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        relative_poses.append({
            "name": img["name"],
            "R": R_rel,
            "t": t_rel,
            "trans": trans_dist,
            "angle": angle,
            "camera_id": img["camera_id"],
        })

    # Sort by name to match image order
    relative_poses.sort(key=lambda x: x["name"])

    # 5. Select poses to render
    if args.pose_indices is not None:
        selected = [(i, relative_poses[i]) for i in args.pose_indices]
    else:
        # 10 evenly spaced
        n = len(relative_poses)
        indices = [int(i * (n - 1) / 9) for i in range(10)]
        selected = [(i, relative_poses[i]) for i in indices]

    print(f"\nSelected {len(selected)} views:")
    for i, rp in selected:
        print(f"  [{i:2d}] {rp['name']}: trans={rp['trans']:.2f}, angle={rp['angle']:.1f}°")

    # 6. Prepare gaussians for render_cpu_alpha format
    # render_cpu_alpha expects world_to_camera transforms
    # Flash3D points are in camera-0 coords, so "world" = camera-0 coords
    # The relative pose IS the world_to_camera transform for each view
    gaussians = {
        "xyz": torch.from_numpy(xyz_scaled).float(),
        "opacity": torch.from_numpy(opacity).reshape(-1, 1).float(),
        "scaling": torch.from_numpy(scaling * scale).float(),  # scale splat sizes too
        "rotation": torch.from_numpy(rotation).float(),
        "features_dc": torch.from_numpy(features_dc).float(),
    }

    # Save gaussians
    gaussians_path = args.output / "gaussians.pt"
    torch.save({"gaussians": gaussians, "metadata": {
        "source": "Flash3D single-image, scaled to COLMAP",
        "scale": scale,
        "ref_image": ref_img["name"],
    }}, gaussians_path)
    print(f"\nSaved scaled gaussians to {gaussians_path}")

    # Save poses as 4x4 world-to-camera matrices
    poses = []
    pose_info = []
    for i, rp in selected:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rp["R"]
        w2c[:3, 3] = rp["t"]
        poses.append(torch.from_numpy(w2c).float())

        cam = cameras[rp["camera_id"]]
        sx = args.render_width / cam["width"]
        sy = args.render_height / cam["height"]
        pose_info.append({
            "name": rp["name"],
            "index": i,
            "trans": rp["trans"],
            "angle": rp["angle"],
            "fx": cam["params"][0] * sx,
            "fy": cam["params"][1] * sy,
            "cx": (cam["params"][2] * sx),
            "cy": (cam["params"][3] * sy),
            "width": args.render_width,
            "height": args.render_height,
        })

    poses_tensor = torch.stack(poses)
    torch.save(poses_tensor, args.output / "render_poses.pt")
    with (args.output / "render_poses.json").open("w") as f:
        json.dump(pose_info, f, indent=2)

    print(f"Saved {len(poses)} poses to {args.output / 'render_poses.pt'}")

    # Print render commands
    print(f"\n=== Render commands ===")
    for pi in pose_info:
        print(f"  {pi['name']}: fx={pi['fx']:.1f}, fy={pi['fy']:.1f}, "
              f"trans={pi['trans']:.2f}, angle={pi['angle']:.1f}°")


if __name__ == "__main__":
    main()
