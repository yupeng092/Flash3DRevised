#!/usr/bin/env python3
"""Convert COLMAP sparse reconstruction to Flash3D gaussians.pt format.

Reads cameras.txt, images.txt, points3D.txt from a COLMAP sparse directory,
converts the 3D points into the gaussian dict expected by render_cpu_alpha.py,
and saves camera poses for multi-view rendering.

Usage:
    python colmap_to_gaussians.py \
        --colmap-dir "D:/Python Project/courtyard/dslr_calibration_undistorted" \
        --output outputs/courtyard_colmap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--colmap-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/courtyard_colmap"))
    parser.add_argument("--scale", type=float, default=0.15, help="Gaussian splat scale")
    parser.add_argument("--opacity", type=float, default=0.95, help="Gaussian opacity")
    return parser.parse_args()


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        cameras[cam_id] = {
            "model": model, "width": width, "height": height, "params": params
        }
    return cameras


def read_images(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    images = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images.append({
            "id": img_id,
            "qvec": np.array([qw, qx, qy, qz], dtype=np.float64),
            "tvec": np.array([tx, ty, tz], dtype=np.float64),
            "camera_id": cam_id,
            "name": name,
        })
        # skip the POINTS2D line
        i += 1
    return images


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz_list = []
    rgb_list = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        xyz_list.append([x, y, z])
        rgb_list.append([r, g, b])
    return np.array(xyz_list, dtype=np.float32), np.array(rgb_list, dtype=np.float32)


def quaternion_to_matrix(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2 * (qy*qy + qz*qz),  2 * (qx*qy - qw*qz),  2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz),      1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy),      2 * (qy*qz + qw*qx),     1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def pose_to_w2c(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion+translation -> 4x4 world-to-camera matrix."""
    R = quaternion_to_matrix(qvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec
    return w2c


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Reading COLMAP files...")
    cameras = read_cameras(args.colmap_dir / "cameras.txt")
    images = read_images(args.colmap_dir / "images.txt")
    xyz, rgb = read_points3d(args.colmap_dir / "points3D.txt")

    print(f"  Cameras: {len(cameras)}")
    print(f"  Images:  {len(images)}")
    print(f"  Points:  {len(xyz)}")
    print(f"  XYZ range: x=[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] "
          f"y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] "
          f"z=[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")

    # Convert points to gaussian format
    n = len(xyz)
    gaussians = {
        "xyz": torch.from_numpy(xyz).float(),
        "opacity": torch.full((n, 1), args.opacity, dtype=torch.float32),
        "scaling": torch.full((n, 3), args.scale, dtype=torch.float32),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float32),
        "features_dc": torch.from_numpy(
            (rgb / 255.0 - 0.5) / SH_C0
        ).float(),
    }

    metadata = {
        "format_version": 2,
        "source": "COLMAP sparse reconstruction",
        "colmap_dir": str(args.colmap_dir.resolve()),
        "num_points": n,
        "num_images": len(images),
        "gaussians_per_pixel": 1,
    }

    gaussians_path = args.output / "gaussians.pt"
    torch.save({"gaussians": gaussians, "metadata": metadata}, gaussians_path)
    print(f"Saved gaussians to {gaussians_path}")

    # Convert camera poses
    poses = []
    pose_info = []
    for img in sorted(images, key=lambda x: x["name"]):
        w2c = pose_to_w2c(img["qvec"], img["tvec"])
        cam = cameras[img["camera_id"]]
        poses.append(torch.from_numpy(w2c).float())
        pose_info.append({
            "name": img["name"],
            "camera_id": img["camera_id"],
            "width": cam["width"],
            "height": cam["height"],
            "fx": cam["params"][0],
            "fy": cam["params"][1],
            "cx": cam["params"][2],
            "cy": cam["params"][3],
        })

    poses_tensor = torch.stack(poses)
    torch.save(poses_tensor, args.output / "render_poses.pt")
    with (args.output / "render_poses.json").open("w") as f:
        json.dump(
            [{"name": p["name"], "fx": p["fx"], "fy": p["fy"],
              "cx": p["cx"], "cy": p["cy"],
              "width": p["width"], "height": p["height"]}
             for p in pose_info],
            f, indent=2
        )
    print(f"Saved {len(poses)} camera poses to {args.output / 'render_poses.pt'}")

    # Print first camera params for render_cpu_alpha.py
    first = pose_info[0]
    scale_x = 384 / first["width"]
    scale_y = 256 / first["height"]
    print(f"\n=== render_cpu_alpha.py 参数 ===")
    print(f"原始相机: {first['width']}x{first['height']}, fx={first['fx']:.1f}")
    print(f"缩放到 384x256: fx={first['fx']*scale_x:.1f}, fy={first['fy']*scale_y:.1f}")
    print(f"")
    print(f"命令示例:")
    print(f"python render_cpu_alpha.py \\")
    print(f"  --gaussians {gaussians_path} \\")
    print(f"  --poses {args.output / 'render_poses.pt'} \\")
    print(f"  --output {args.output / 'rendered'} \\")
    print(f"  --height 256 --width 384 \\")
    print(f"  --fx {first['fx']*scale_x:.1f} --fy {first['fy']*scale_y:.1f} \\")
    print(f"  --pose-index 0 5 10 15 20 25 30 35 \\")
    print(f"  --supersample 2 --sharpen 0.3 --keep-ratio 1.0")


if __name__ == "__main__":
    main()
