#!/usr/bin/env python3
"""Generate a tiny RE10K-format synthetic dataset from existing courtyard images.

This creates the minimum data contract that ``datasets/re10k.py`` and
``train.py`` need to run an end-to-end CPU smoke step, without downloading the
real RealEstate10K dataset (which requires YouTube + ffmpeg + days).

It uses the multi-view courtyard render images already present under
``outputs/courtyard_render/rgb/`` and synthesises:
  - ``{train,test}.pickle.gz`` sequence metadata (timestamps/intrinsics/poses)
  - ``{train,test}/<seq>/<timestamp>.jpg`` images
  - ``pcl.{train,test}.tar`` sparse COLMAP point clouds (xys/p3D_ids/xyz)
  - ``valid_seq_ids.train.pickle.gz``

The camera trajectory is a synthetic horizontal arc; intrinsics are a reasonable
guess for the courtyard renders.  This is NOT real RE10K data and cannot train a
useful model — it only validates the training pipeline runs end-to-end on CPU.
"""
from __future__ import annotations

import gzip
import math
import os
import pickle
import tarfile
import io
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "outputs" / "courtyard_render" / "rgb"
DATA_DIR = PROJECT_ROOT / "data" / "RealEstate10K"

# Target image size (must be multiples of 32 for the ResNet decoder).
H, W = 128, 192
# Synthetic intrinsics (normalised by image size, as RE10K stores them).
# focal ~0.7*W, principal point at center.
FX_NORM, FY_NORM = 0.7, 0.7
CX_NORM, CY_NORM = 0.5, 0.5
# Number of source frames per sequence.
FRAMES_PER_SEQ = 5


def load_resize_images() -> list[Image.Image]:
    images = []
    for i in range(10):
        p = SRC_DIR / f"view_{i:03d}.png"
        if not p.is_file():
            raise FileNotFoundError(p)
        img = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        images.append(img)
    return images


def make_trajectory(n: int) -> np.ndarray:
    """Return n camera-to-world poses along a horizontal arc (looking forward)."""
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)  # 0..1
        angle = (t - 0.5) * 0.3  # ~+-0.15 rad yaw
        x = (t - 0.5) * 0.6      # horizontal translation
        c, s = math.cos(angle), math.sin(angle)
        # world-to-camera: rotate by -angle around Y, translate by -x.
        w2c = np.eye(4, dtype=np.float32)
        w2c[0, 0], w2c[0, 2] = c, -s
        w2c[2, 0], w2c[2, 2] = s, c
        w2c[0, 3] = -x
        poses.append(w2c)
    return np.stack(poses)


def make_intrinsics() -> np.ndarray:
    """RE10K stores intrinsics as [fx, fy, cx, cy] normalised by image size."""
    return np.array([FX_NORM, FY_NORM, CX_NORM, CY_NORM], dtype=np.float32)


def make_sparse_pcl(poses: np.ndarray, n_points: int = 80) -> dict:
    """Build a sparse COLMAP-style point cloud visible from all frames.

    Points are scattered in front of the cameras; each frame observes all of
    them (p3D_ids all valid) so ``get_sparse_depth`` has data to work with.
    """
    rng = np.random.default_rng(42)
    # Random 3D points in a volume in front of the first camera (z in [2, 6]).
    xyz = np.column_stack([
        rng.uniform(-1.5, 1.5, n_points),
        rng.uniform(-1.0, 1.0, n_points),
        rng.uniform(2.0, 6.0, n_points),
    ]).astype(np.float32)

    xys_all, p3D_ids_all = [], []
    K = np.array([[FX_NORM * W, 0, CX_NORM * W],
                  [0, FY_NORM * H, CY_NORM * H],
                  [0, 0, 1]], dtype=np.float32)
    for w2c in poses:
        xyz_h = np.hstack([xyz, np.ones((n_points, 1), dtype=np.float32)])
        cam_pts = (w2c @ xyz_h.T).T[:, :3]
        # Project.
        proj = (K @ cam_pts.T).T
        pix = proj[:, :2] / np.clip(proj[:, 2:3], 1e-3, None)
        xys = pix.astype(np.float32)
        p3D_ids = np.arange(n_points, dtype=np.int64)
        # Mark points behind camera as unobserved.
        visible = cam_pts[:, 2] > 0.1
        p3D_ids = np.where(visible, p3D_ids, -1)
        xys_all.append(xys)
        p3D_ids_all.append(p3D_ids)
    return {"xys": xys_all, "p3D_ids": p3D_ids_all, "xyz": xyz}


def write_pcl_tar(entries: list[tuple[str, np.ndarray]], split: str, tar_path: Path) -> None:
    """Write all sequence pcl entries into one tar (accumulative, not overwriting).

    entries: list of (seq_key, poses).
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale tar so re-runs don't accumulate duplicates.
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w") as tar:
        for seq_key, poses in entries:
            pcl = make_sparse_pcl(poses)
            blob = pickle.dumps(pcl)
            inner = f"pcl.{split}/{seq_key}.pickle.gz"
            data = gzip.compress(blob)
            info = tarfile.TarInfo(name=inner)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def build_split(split: str, images: list[Image.Image], seq_offset: int) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """Write images + metadata for one sequence; return (seq_data, pcl_entries).

    pcl_entries is a list of (seq_key, poses) to be written into the split's
    pcl tar later (all sequences in one tar, not one-per-file).
    """
    split_dir = DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    seq_data = {}
    seq_key = f"courtyard_{split}_{seq_offset:02d}"
    poses = make_trajectory(FRAMES_PER_SEQ)
    intrinsics = np.stack([make_intrinsics()] * FRAMES_PER_SEQ)
    timestamps = [f"{1000 * (i + 1):013d}" for i in range(FRAMES_PER_SEQ)]

    # Write images.
    seq_img_dir = split_dir / seq_key
    seq_img_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(timestamps):
        idx = (seq_offset + i) % len(images)
        images[idx].save(seq_img_dir / f"{ts}.jpg", quality=90)

    seq_data[seq_key] = {"timestamps": timestamps, "intrinsics": intrinsics, "poses": poses}
    return seq_data, [(seq_key, poses)]


def split_offset():
    return 0


def main() -> None:
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(f"Courtyard images not found at {SRC_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images = load_resize_images()
    print(f"[setup] {len(images)} source images ({W}x{H})")

    # Build 2 train sequences + 1 test sequence.  Collect pcl entries so each
    # split's tar contains ALL its sequences (not just the last one).
    train_data = {}
    train_pcl_entries: list[tuple[str, np.ndarray]] = []
    for i in range(2):
        d, entries = build_split("train", images, seq_offset=i * FRAMES_PER_SEQ)
        train_data.update(d)
        train_pcl_entries.extend(entries)
    test_data, test_pcl_entries = build_split("test", images, seq_offset=2 * FRAMES_PER_SEQ)

    # Write COLMAP pcl tars (one per split, containing all sequences).
    write_pcl_tar(train_pcl_entries, "train", DATA_DIR / "pcl.train.tar")
    write_pcl_tar(test_pcl_entries, "test", DATA_DIR / "pcl.test.tar")
    print(f"[write] pcl.train.tar ({len(train_pcl_entries)} seqs), pcl.test.tar ({len(test_pcl_entries)} seqs)")

    # Write metadata pickles.
    with gzip.open(DATA_DIR / "train.pickle.gz", "wb") as f:
        pickle.dump(train_data, f)
    with gzip.open(DATA_DIR / "test.pickle.gz", "wb") as f:
        pickle.dump(test_data, f)
    print(f"[write] train.pickle.gz ({len(train_data)} seqs), test.pickle.gz ({len(test_data)} seqs)")

    # Write valid_seq_ids (all train sequences valid).
    valid = {k: list(range(FRAMES_PER_SEQ)) for k in train_data}
    with gzip.open(DATA_DIR / "valid_seq_ids.train.pickle.gz", "wb") as f:
        pickle.dump(valid, f)
    print(f"[write] valid_seq_ids.train.pickle.gz")

    # Write test/val split files compatible with re10k.py's _load_split_indices.
    # Format: "seqkey src_idx tgt5 tgt10 tgtrandom"
    split_lines = []
    for k, d in test_data.items():
        n = len(d["timestamps"])
        split_lines.append(f"{k} 0 {min(1, n-1)} {min(2, n-1)} {min(3, n-1)}")
    split_content = "\n".join(split_lines) + "\n"
    split_dir = PROJECT_ROOT / "splits" / "re10k_mine_filtered"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("test_files.txt", "val_files.txt"):
        p = split_dir / name
        p.write_text(split_content, encoding="utf-8")
        print(f"[write] {p}")

    print("\n=== summary ===")
    for split, data in [("train", train_data), ("test", test_data)]:
        for k, d in data.items():
            print(f"  {split}/{k}: {len(d['timestamps'])} frames, poses {d['poses'].shape}")
    print(f"\nData written to {DATA_DIR}")
    print("Ready for: python train.py +experiment=layered_re10k_cpu_debug_v1 "
          "dataset.data_path=data/RealEstate10K optimiser.num_epochs=1 "
          "model.backbone.weights_init=scratch model.depth.encoder=vits")


if __name__ == "__main__":
    main()
