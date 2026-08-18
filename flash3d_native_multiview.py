#!/usr/bin/env python3
"""Test Flash3D native multi-view rendering using COLMAP poses.

This constructs the exact `inputs` dict that Flash3D's GaussianPredictor
expects (matching datasets/re10k.py's format), but with COLMAP camera
poses from the courtyard dataset instead of RE10K video frames.

It calls GaussianPredictor.forward() directly, which internally:
  1. Runs UniDepth on the source image -> depth + intrinsics
  2. Backprojects depth -> 3D gaussians
  3. Uses the provided T_c2w poses to render each target frame

This is the NATIVE Flash3D multi-view pipeline - same code path as
evaluate.py, just with courtyard COLMAP data instead of RE10K.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from hydra import compose, initialize_config_dir
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import GaussianPredictor


SH_C0 = 0.28209479177387814


def quat_to_rot(q):
    """COLMAP quaternion (w,x,y,z) -> 3x3 rotation matrix."""
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
        img_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]
        # Parse POINTS2D line: (X, Y, POINT3D_ID) triples
        points2d_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        xys = []
        p3d_ids = []
        if points2d_line and not points2d_line.startswith("#"):
            nums = points2d_line.split()
            for j in range(0, len(nums) - 2, 3):
                xys.append([float(nums[j]), float(nums[j+1])])
                p3d_ids.append(int(nums[j+2]))
        images.append({
            "id": img_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "xys": np.array(xys) if xys else np.zeros((0, 2)),
            "p3d_ids": np.array(p3d_ids) if p3d_ids else np.zeros((0,), dtype=np.int64),
        })
    return images


def read_colmap_points3d(path):
    """Read COLMAP points3D.txt -> dict with id->xyz mapping and array."""
    pts = {}
    xyz_list = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pid = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        pts[pid] = np.array([x, y, z], dtype=np.float64)
        xyz_list.append([x, y, z])
    return pts, np.array(xyz_list) if xyz_list else np.zeros((0, 3))


def get_sparse_depth(src_img, points3d_map, orig_w, orig_h, render_w, render_h, device):
    """Extract sparse depth from COLMAP for the source image.

    Returns [N, 3] tensor: (x_norm, y_norm, depth) where x/y are in [-1, 1]
    (matching Flash3D's grid_sample convention) and depth is in COLMAP units.
    """
    xys = src_img["xys"]
    p3d_ids = src_img["p3d_ids"]

    # Filter visible points (p3d_id != -1)
    visible = p3d_ids != -1
    xys = xys[visible]
    p3d_ids = p3d_ids[visible]

    if len(p3d_ids) == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    # Get 3D points
    xyz = np.array([points3d_map[pid] for pid in p3d_ids if pid in points3d_map])
    if len(xyz) == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    # Project to camera using source pose
    # COLMAP: P_cam = R @ P_world + t
    R = quat_to_rot(src_img["qvec"])
    t = np.array(src_img["tvec"])
    xyz_cam = (R @ xyz.T).T + t  # [N, 3]
    depths = xyz_cam[:, 2]

    # Scale 2D pixel coords from original image resolution to render resolution
    # then normalize to [-1, 1] (matching get_sparse_depth in data.py)
    xys_scaled = (xys / np.array([[orig_w, orig_h]]) - 0.5) * 2

    xyd = np.concatenate([xys_scaled, depths[:, None]], axis=1)
    return torch.from_numpy(xyd).to(torch.float32).to(device)


def read_colmap_cameras(path):
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = {
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(x) for x in parts[4:]],
        }
    return cameras


def load_image(path, height, width, pad):
    """Load and preprocess image matching RE10K format: RGB [0,1], resize, zero-pad."""
    image = Image.open(path).convert("RGB")
    image = TVF.resize(image, [height, width], interpolation=InterpolationMode.LANCZOS)
    color = TVF.to_tensor(image).unsqueeze(0)  # [1, 3, H, W]
    color_aug = F.pad(color, (pad, pad, pad, pad)) if pad else color
    return color, color_aug


def make_flash3d_inputs(
    src_image_path,
    src_pose,
    tgt_poses,
    camera_params,
    height,
    width,
    pad,
    device,
):
    """Construct the inputs dict matching datasets/re10k.py format.

    Keys required by GaussianPredictor.forward():
      - ("color", 0, 0): source image [B, 3, H, W]
      - ("color_aug", 0, 0): padded source image
      - ("K_src", 0): source intrinsics [B, 3, 3]
      - ("K_tgt", frame_id): target intrinsics [B, 3, 3] for each frame
      - ("T_c2w", frame_id): camera-to-world pose [B, 4, 4] for each frame
      - ("T_w2c", frame_id): world-to-camera pose (inverse of T_c2w)
      - "target_frame_ids": list of target frame ids
    """
    B = 1

    # Load source image
    color, color_aug = load_image(src_image_path, height, width, pad)
    color = color.to(device)
    color_aug = color_aug.to(device)

    # Scale intrinsics from original resolution to render resolution
    orig_w = camera_params["width"]
    orig_h = camera_params["height"]
    sx = width / orig_w
    sy = height / orig_h
    fx = camera_params["params"][0] * sx
    fy = camera_params["params"][1] * sy
    cx = camera_params["params"][2] * sx
    cy = camera_params["params"][3] * sy

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]

    inputs = {}
    inputs[("color", 0, 0)] = color
    inputs[("color_aug", 0, 0)] = color_aug
    inputs[("K_src", 0)] = K
    inputs[("K_tgt", 0)] = K  # source camera = target for frame 0

    # Source pose (frame 0)
    R_src = quat_to_rot(src_pose["qvec"])
    t_src = np.array(src_pose["tvec"])
    # COLMAP gives world-to-camera: P_cam = R @ P_world + t
    # Flash3D uses camera-to-world: T_c2w
    # R_c2w = R^T, t_c2w = -R^T @ t
    R_c2w = R_src.T
    t_c2w = -R_c2w @ t_src
    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    T_c2w_tensor = torch.from_numpy(T_c2w).float().unsqueeze(0).to(device)
    inputs[("T_c2w", 0)] = T_c2w_tensor
    inputs[("T_w2c", 0)] = torch.linalg.inv(T_c2w_tensor)

    # Target poses
    target_frame_ids = []
    for i, tgt_pose in enumerate(tgt_poses, start=1):
        frame_id = i
        target_frame_ids.append(frame_id)

        R_tgt = quat_to_rot(tgt_pose["qvec"])
        t_tgt = np.array(tgt_pose["tvec"])
        R_c2w_tgt = R_tgt.T
        t_c2w_tgt = -R_c2w_tgt @ t_tgt
        T_c2w_tgt = np.eye(4, dtype=np.float64)
        T_c2w_tgt[:3, :3] = R_c2w_tgt
        T_c2w_tgt[:3, 3] = t_c2w_tgt
        T_c2w_tensor = torch.from_numpy(T_c2w_tgt).float().unsqueeze(0).to(device)

        inputs[("T_c2w", frame_id)] = T_c2w_tensor
        inputs[("T_w2c", frame_id)] = torch.linalg.inv(T_c2w_tensor)
        inputs[("K_tgt", frame_id)] = K  # same camera for all

    inputs["target_frame_ids"] = target_frame_ids
    return inputs


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Source image")
    p.add_argument("--colmap-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/flash3d_native_multiview"))
    p.add_argument("--ref-image", default="DSC_0286")
    p.add_argument("--num-views", type=int, default=10)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=384)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rgb").mkdir(exist_ok=True)

    # Load config
    print("Loading Flash3D config...")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[
            "+experiment=layered_re10k",
            "model.depth.version=v1",
        ])

    # Override for CPU inference
    cfg.model.gaussian_rendering = True  # MUST be True for rendering
    cfg.model.randomise_bg_colour = False
    cfg.model.renderer_backend = "torch"  # Use portable renderer
    cfg.model.gauss_novel_frames = list(range(1, args.num_views))
    cfg.train.use_gt_poses = True
    cfg.train.scale_pose_by_depth = True
    cfg.dataset.height = args.height
    cfg.dataset.width = args.width
    cfg.dataset.pad_border_aug = 32
    cfg.dataset.name = "re10k"
    cfg.dataset.scale_pose_by_depth = True
    cfg.dataset.znear = 0.01
    cfg.dataset.zfar = 100.0
    cfg.data_loader.batch_size = 1  # Match inference batch

    device = torch.device("cpu")

    # Load model
    print("Loading Flash3D model...")
    model = GaussianPredictor(cfg).to(device)
    model.load_model(args.checkpoint, device="cpu")
    model.set_eval()
    print(f"  Model loaded, renderer_backend={cfg.model.renderer_backend}")

    # Read COLMAP
    print("Reading COLMAP data...")
    images = read_colmap_images(args.colmap_dir / "images.txt")
    cameras = read_colmap_cameras(args.colmap_dir / "cameras.txt")
    points3d_map, _ = read_colmap_points3d(args.colmap_dir / "points3D.txt")
    print(f"  {len(images)} images, {len(points3d_map)} 3D points")

    # Find reference image
    ref_img = None
    for img in images:
        if args.ref_image in img["name"]:
            ref_img = img
            break
    if ref_img is None:
        raise ValueError(f"Reference image {args.ref_image} not found")

    # Sort images by angle relative to reference
    R0 = quat_to_rot(ref_img["qvec"])
    t0 = np.array(ref_img["tvec"])
    angles = []
    for img in images:
        Ri = quat_to_rot(img["qvec"])
        R_rel = Ri @ R0.T
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        angles.append(angle)

    # Sort by angle, take closest N (including ref)
    sorted_indices = np.argsort(angles)
    selected = [images[i] for i in sorted_indices[:args.num_views]]
    print(f"Selected {len(selected)} views (sorted by angle):")
    for i, img in enumerate(selected):
        print(f"  [{i}] {img['name']}: angle={angles[sorted_indices[i]]:.1f}°")

    # Camera params (use ref image's camera)
    cam = cameras[ref_img["camera_id"]]

    # Construct inputs
    src_pose = selected[0]
    tgt_poses = selected[1:]

    print(f"\nConstructing Flash3D inputs...")
    print(f"  Source: {src_pose['name']}")
    print(f"  Targets: {len(tgt_poses)} views")

    inputs = make_flash3d_inputs(
        src_image_path=args.image,
        src_pose=src_pose,
        tgt_poses=tgt_poses,
        camera_params=cam,
        height=args.height,
        width=args.width,
        pad=cfg.dataset.pad_border_aug,
        device=device,
    )

    # Extract COLMAP sparse depth for scale alignment (RANSAC)
    print("Extracting COLMAP sparse depth for RANSAC scale alignment...")
    sparse_depth = get_sparse_depth(
        src_pose, points3d_map, cam["width"], cam["height"],
        args.width, args.height, device
    )
    print(f"  {sparse_depth.shape[0]} sparse depth points")
    # model.py expects [B, N, 3]: sparse_depth[k] gives [N, 3] for batch k
    inputs[("depth_sparse", 0)] = sparse_depth.unsqueeze(0)  # [1, N, 3]

    print(f"  Input keys: {len(inputs)} entries")
    print(f"  Source image shape: {inputs[('color', 0, 0)].shape}")
    print(f"  Target frame IDs: {inputs['target_frame_ids']}")

    # Run Flash3D forward (native multi-view rendering)
    print(f"\nRunning Flash3D native forward pass...")
    with torch.no_grad():
        outputs = model(inputs)

    print(f"\nOutput keys: {len(outputs)} entries")

    # Save gaussians.pt for render_cpu_alpha.py
    print(f"\nSaving gaussians.pt for render_cpu_alpha.py...")
    # Flatten Flash3D outputs to render_cpu_alpha format
    means = outputs["gauss_means"][:, :3, :]  # [B*gpp, 3, N]
    xyz_flat = means.permute(0, 2, 1).reshape(-1, 3).contiguous()
    opacity_flat = outputs["gauss_opacity"].permute(0, 2, 3, 1).reshape(-1, 1).contiguous()
    scaling_flat = outputs["gauss_scaling"].permute(0, 2, 3, 1).reshape(-1, 3).contiguous()
    rotation_flat = outputs["gauss_rotation"].permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
    features_dc_flat = outputs["gauss_features_dc"].permute(0, 2, 3, 1).reshape(-1, 3).contiguous()
    if "gauss_features_rest" in outputs:
        features_rest_flat = outputs["gauss_features_rest"].permute(0, 2, 3, 1).reshape(-1, 9).contiguous()
    else:
        features_rest_flat = None

    saved_gaussians = {
        "xyz": xyz_flat.float().cpu(),
        "opacity": opacity_flat.float().cpu(),
        "scaling": scaling_flat.float().cpu(),
        "rotation": rotation_flat.float().cpu(),
        "features_dc": features_dc_flat.float().cpu(),
    }
    if features_rest_flat is not None:
        saved_gaussians["features_rest"] = features_rest_flat.float().cpu()

    saved_metadata = {
        "format_version": 1,
        "coordinate_system": "Flash3D source-camera coordinates: x right, y down, z forward",
        "input_image": str(args.image.resolve()),
        "input_size_hw": [args.height, args.width],
        "padded_size_hw": [args.height + 2 * cfg.dataset.pad_border_aug, args.width + 2 * cfg.dataset.pad_border_aug],
        "gaussians_per_pixel": int(cfg.model.gaussians_per_pixel),
        "max_sh_degree": int(cfg.model.max_sh_degree),
        "checkpoint": str(args.checkpoint.resolve()),
        "scale_applied": float(outputs.get(("depth_scale", 0), torch.tensor([1.0]))[0].item()),
    }
    gaussians_path = args.output / "gaussians.pt"
    torch.save({"gaussians": saved_gaussians, "metadata": saved_metadata}, gaussians_path)
    print(f"  Saved {len(xyz_flat)} gaussians to {gaussians_path}")

    # Save render_poses.pt (world-to-camera matrices for each view)
    # render_cpu_alpha.py expects [N, 4, 4] w2c transforms
    # Flash3D point cloud is in camera-0 coords, so "world" = camera-0
    # The relative pose (cam_T_cam 0->i) IS the w2c transform for view i
    print(f"\nSaving render_poses.pt...")
    poses_list = []
    for frame_id in [0] + inputs["target_frame_ids"]:
        if frame_id == 0:
            w2c = torch.eye(4, dtype=torch.float32)
        else:
            cam_T = outputs.get(("cam_T_cam", 0, frame_id))
            if cam_T is None:
                continue
            w2c = cam_T[0].float().cpu()
        poses_list.append(w2c)
    poses_tensor = torch.stack(poses_list)
    torch.save(poses_tensor, args.output / "render_poses.pt")
    print(f"  Saved {len(poses_list)} poses to {args.output / 'render_poses.pt'}")

    # Print render_cpu_alpha.py command
    print(f"\n=== render_cpu_alpha.py command ===")
    print(f"python render_cpu_alpha.py \\")
    print(f"  --gaussians {gaussians_path} \\")
    print(f"  --poses {args.output / 'render_poses.pt'} \\")
    print(f"  --output {args.output / 'rendered'} \\")
    print(f"  --height 256 --width 384 \\")
    print(f"  --fx 211.0 --fy 211.1 --cx 192 --cy 128 \\")
    print(f"  --pose-index {' '.join(str(i) for i in range(len(poses_list)))} \\")
    print(f"  --supersample 1 --keep-ratio 1.0 --min-opacity 0.0 --threads 28")

    # Save rendered images (from Flash3D native forward, for reference)
    print(f"\nSaving Flash3D native rendered images (for reference)...")
    all_frame_ids = [0] + inputs["target_frame_ids"]
    for frame_id in all_frame_ids:
        key = ("color_gauss", frame_id, 0)
        if key not in outputs:
            print(f"  frame {frame_id}: NO OUTPUT")
            continue
        rgb = outputs[key][0]  # [3, H, W]
        rgb = rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        rgb_uint8 = (rgb * 255 + 0.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_uint8, "RGB").save(args.output / "rgb" / f"view_{frame_id:03d}.png")

        # Also save depth if available
        depth_key = ("depth_gauss", frame_id, 0)
        if depth_key in outputs:
            depth = outputs[depth_key][0].cpu().numpy()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_img = (depth_norm * 255).astype(np.uint8)
            Image.fromarray(depth_img, "L").save(args.output / f"depth_{frame_id:03d}.png")

        pose_name = selected[frame_id]["name"] if frame_id < len(selected) else f"frame_{frame_id}"
        print(f"  view_{frame_id:03d} ({pose_name}): saved {rgb_uint8.shape[1]}x{rgb_uint8.shape[0]}")

    # Save metadata
    metadata = {
        "source_image": str(args.image.resolve()),
        "ref_image": args.ref_image,
        "num_views": args.num_views,
        "resolution": [args.width, args.height],
        "renderer": "Flash3D native (GaussianPredictor.forward + render_predicted_torch)",
        "views": [{"name": img["name"], "angle": float(angles[sorted_indices[i]])}
                  for i, img in enumerate(selected)],
    }
    with (args.output / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved {len(all_frame_ids)} views to {args.output / 'rgb'}")


if __name__ == "__main__":
    main()
