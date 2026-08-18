#!/usr/bin/env python3
"""Scale benchmark_cpu gaussians.pt to COLMAP coordinates and generate poses.

Takes the raw Flash3D point cloud from benchmark_cpu.py and:
1. Estimates scale factor from depth medians (Flash3D vs COLMAP)
2. Scales xyz and scaling parameters
3. Computes relative COLMAP poses (camera 0 -> camera i)
4. Saves gaussians_scaled.pt + render_poses.pt for render_cpu_alpha.py
"""
import torch
import numpy as np
import json
from pathlib import Path

def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)

def main():
    benchmark_dir = Path("outputs/benchmark_inference")
    colmap_dir = Path(r"D:\Python Project\courtyard\dslr_calibration_undistorted")
    output_dir = Path("outputs/benchmark_scaled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark_cpu gaussians
    print("Loading benchmark_cpu gaussians...")
    data = torch.load(benchmark_dir / "gaussians.pt", map_location="cpu", weights_only=False)
    g = data["gaussians"]
    print(f"  {g['xyz'].shape[0]} gaussians")
    xyz = g["xyz"].numpy()
    flash3d_depth_median = np.median(xyz[:, 2])
    print(f"  Flash3D depth median: {flash3d_depth_median:.2f}")

    # 2. Read COLMAP to estimate scale
    print("Reading COLMAP...")
    # cameras
    cameras = {}
    for line in (colmap_dir / "cameras.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        cameras[int(parts[0])] = {"width": int(parts[2]), "height": int(parts[3]),
                                   "params": [float(x) for x in parts[4:]]}
    # images
    images = []
    lines = (colmap_dir / "images.txt").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line or line.startswith("#"): continue
        parts = line.split()
        images.append({"id": int(parts[0]), "qvec": [float(x) for x in parts[1:5]],
                       "tvec": [float(x) for x in parts[5:8]], "camera_id": int(parts[8]),
                       "name": parts[9]})
        i += 1
    # points3D
    colmap_pts = []
    for line in (colmap_dir / "points3D.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        colmap_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    colmap_pts = np.array(colmap_pts)

    # Find ref image
    ref_img = [img for img in images if "DSC_0286" in img["name"]][0]
    R0 = quat_to_rot(ref_img["qvec"])
    t0 = np.array(ref_img["tvec"])

    # Scale estimation
    pts_cam0 = (R0 @ colmap_pts.T).T + t0
    colmap_depths = pts_cam0[:, 2]
    colmap_depths = colmap_depths[colmap_depths > 0]
    colmap_depth_median = np.median(colmap_depths)
    scale = colmap_depth_median / flash3d_depth_median
    print(f"  COLMAP depth median: {colmap_depth_median:.2f}")
    print(f"  Scale factor: {scale:.4f}")

    # 3. Scale gaussians
    g_scaled = {k: v.clone() for k, v in g.items()}
    g_scaled["xyz"] = g["xyz"] * scale
    g_scaled["scaling"] = g["scaling"] * scale

    metadata = data.get("metadata", {})
    metadata["scale_applied"] = scale
    torch.save({"gaussians": g_scaled, "metadata": metadata}, output_dir / "gaussians.pt")
    print(f"  Saved scaled gaussians to {output_dir / 'gaussians.pt'}")

    # 4. Select 10 nearest-angle views and compute relative poses
    angles = []
    for img in images:
        Ri = quat_to_rot(img["qvec"])
        R_rel = Ri @ R0.T
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        angles.append(angle)

    sorted_idx = np.argsort(angles)
    selected = [images[i] for i in sorted_idx[:10]]
    print(f"\nSelected 10 views:")
    for i, img in enumerate(selected):
        print(f"  [{i}] {img['name']}: angle={angles[sorted_idx[i]]:.1f}°")

    # Compute relative w2c poses (camera-0 coords = "world")
    poses = []
    for img in selected:
        Ri = quat_to_rot(img["qvec"])
        ti = np.array(img["tvec"])
        R_rel = Ri @ R0.T
        t_rel = ti - R_rel @ t0
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_rel
        w2c[:3, 3] = t_rel
        poses.append(torch.from_numpy(w2c).float())

    poses_tensor = torch.stack(poses)
    torch.save(poses_tensor, output_dir / "render_poses.pt")
    print(f"  Saved {len(poses)} poses to {output_dir / 'render_poses.pt'}")

    # Camera intrinsics scaled to 384x256
    cam = cameras[ref_img["camera_id"]]
    sx = 384 / cam["width"]
    sy = 256 / cam["height"]
    fx = cam["params"][0] * sx
    fy = cam["params"][1] * sy
    print(f"\nRender params: fx={fx:.1f}, fy={fy:.1f}, cx=192, cy=128")

if __name__ == "__main__":
    main()
