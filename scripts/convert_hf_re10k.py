#!/usr/bin/env python3
"""Convert HuggingFace Hualingchu/RealEstate10K_test .torch files to Flash3D format.

Downloads a few .torch files (each ~100MB, ~17 sequences with real RE10K images
+ poses), extracts real images and camera parameters, and writes a Flash3D-
compatible dataset under data/RealEstate10K/:

  {train,test}.pickle.gz          - sequence metadata (timestamps/intrinsics/poses)
  {train,test}/{seq}/{ts}.jpg     - real RE10K images (resized to HxW)
  pcl.{train,test}.tar            - synthetic COLMAP sparse points (real poses)
  valid_seq_ids.train.pickle.gz   - valid sequence filter

The COLMAP point cloud is synthetic (points scattered in front of cameras) but
uses the REAL camera poses, so scale_pose_by_depth has realistic geometry to
work with.  This is NOT a substitute for the full RE10K COLMAP cache, but is
sufficient for CPU training validation.
"""
from __future__ import annotations

import gzip
import io
import math
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATA_DIR = PROJECT_ROOT / "data" / "RealEstate10K"
H, W = 128, 192  # target image size (multiples of 32 for ResNet decoder)
NUM_TORCH_FILES = 10  # download 10 .torch files (~1GB, ~170 sequences)
FRAMES_PER_SEQ = 15   # downsample to 15 frames per sequence for CPU training
NUM_TRAIN_SEQS = 20   # pick 20 sequences for train (300 frames)
NUM_TEST_SEQS = 4     # pick 4 sequences for test (60 frames)


def download_torch_files(n: int) -> list[Path]:
    """Download n .torch files from HuggingFace."""
    from huggingface_hub import hf_hub_download
    raw_dir = PROJECT_ROOT / "data" / "re10k_hf_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        fname = f"test/{i:06d}.torch"
        local = raw_dir / "test" / f"{i:06d}.torch"
        if local.is_file() and local.stat().st_size > 1e6:
            print(f"  [skip] {fname} already exists ({local.stat().st_size / 1e6:.0f} MB)")
            paths.append(local)
            continue
        print(f"  [download] {fname} ...")
        p = hf_hub_download("Hualingchu/RealEstate10K_test", fname,
                            repo_type="dataset", local_dir=str(raw_dir))
        paths.append(Path(p))
        print(f"    done ({Path(p).stat().st_size / 1e6:.0f} MB)")
    return paths


def load_sequences(paths: list[Path]) -> list[dict]:
    """Load all sequences from .torch files."""
    seqs = []
    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        for item in data:
            if isinstance(item, dict) and "key" in item and "images" in item:
                seqs.append(item)
    print(f"  loaded {len(seqs)} sequences from {len(paths)} files")
    return seqs


def pick_sequences(seqs: list[dict], n: int) -> list[dict]:
    """Pick n sequences with the most frames and good camera motion."""
    scored = []
    for s in seqs:
        n_frames = len(s["images"])
        cams = s["cameras"]  # (N, 18)
        # Score by camera translation magnitude (more motion = better training)
        w2c = cams[:, 6:18].reshape(-1, 3, 4)
        trans = w2c[:, :, 3]  # (N, 3)
        motion = float(trans.std(dim=0).sum())  # total translation variance
        scored.append((motion, n_frames, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [s for _, _, s in scored[:n]]
    for i, s in enumerate(picked):
        print(f"    picked[{i}]: key={s['key']} frames={len(s['images'])} motion={scored[i][0]:.4f}")
    return picked


def extract_frames(seq: dict, n_frames: int) -> tuple[list, np.ndarray, np.ndarray]:
    """Downsample to n_frames, decode images, extract intrinsics + poses.

    Returns: (list_of_PIL_images, intrinsics_array, poses_array)
    - intrinsics: (n, 4) [fx, fy, cx, cy] normalized
    - poses: (n, 3, 4) world-to-camera
    """
    total = len(seq["images"])
    if total <= n_frames:
        indices = list(range(total))
    else:
        # Evenly sample
        indices = list(range(0, total, max(1, total // n_frames)))[:n_frames]

    cams = seq["cameras"][indices]  # (n, 18)
    intrinsics = cams[:, :4].numpy().astype(np.float32)  # [fx, fy, cx, cy]
    poses = cams[:, 6:18].reshape(-1, 3, 4).numpy().astype(np.float32)  # w2c 3x4

    images = []
    for idx in indices:
        img_bytes = bytes(seq["images"][idx].tolist())
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((W, H), Image.LANCZOS)
        images.append(img)
    return images, intrinsics, poses


def make_sparse_pcl(poses: np.ndarray, n_points: int = 100) -> dict:
    """Build synthetic COLMAP sparse points using real camera poses."""
    rng = np.random.default_rng(42)
    xyz = np.column_stack([
        rng.uniform(-2.0, 2.0, n_points),
        rng.uniform(-1.5, 1.5, n_points),
        rng.uniform(2.0, 8.0, n_points),
    ]).astype(np.float32)

    # Use first camera's intrinsics to project
    xys_all, p3D_ids_all = [], []
    fx, fy, cx, cy = 0.4725 * W, 0.8400 * H, 0.5 * W, 0.5 * H  # typical RE10K normalized
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    for w2c_34 in poses:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = w2c_34
        xyz_h = np.hstack([xyz, np.ones((n_points, 1), dtype=np.float32)])
        cam_pts = (w2c @ xyz_h.T).T[:, :3]
        proj = (K @ cam_pts.T).T
        pix = proj[:, :2] / np.clip(proj[:, 2:3], 1e-3, None)
        xys = pix.astype(np.float32)
        p3D_ids = np.arange(n_points, dtype=np.int64)
        visible = cam_pts[:, 2] > 0.1
        p3D_ids = np.where(visible, p3D_ids, -1)
        xys_all.append(xys)
        p3D_ids_all.append(p3D_ids)
    return {"xys": xys_all, "p3D_ids": p3D_ids_all, "xyz": xyz}


def write_pcl_tar(entries, split, tar_path):
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w") as tar:
        for seq_key, poses in entries:
            pcl = make_sparse_pcl(poses)
            blob = pickle.dumps(pcl)
            data = gzip.compress(blob)
            info = tarfile.TarInfo(name=f"pcl.{split}/{seq_key}.pickle.gz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def build_split(split: str, seqs: list[dict]) -> tuple[dict, list]:
    """Write images + collect metadata for one split."""
    split_dir = DATA_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    seq_data = {}
    pcl_entries = []
    for seq in seqs:
        key = seq["key"]
        images, intrinsics, poses = extract_frames(seq, FRAMES_PER_SEQ)
        timestamps = [f"{1000 * (i + 1):013d}" for i in range(len(images))]

        # Write images
        img_dir = split_dir / key
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, ts in enumerate(timestamps):
            images[i].save(img_dir / f"{ts}.jpg", quality=90)

        seq_data[key] = {
            "timestamps": timestamps,
            "intrinsics": intrinsics,
            "poses": poses,  # w2c 3x4, Flash3D's data_to_c2w will invert
        }
        pcl_entries.append((key, poses))
    return seq_data, pcl_entries


def main():
    print("=== Step 1: Download .torch files from HuggingFace ===")
    paths = download_torch_files(NUM_TORCH_FILES)

    print("\n=== Step 2: Load sequences ===")
    seqs = load_sequences(paths)

    print(f"\n=== Step 3: Pick {NUM_TRAIN_SEQS} train + {NUM_TEST_SEQS} test sequences ===")
    train_seqs = pick_sequences(seqs, NUM_TRAIN_SEQS)
    train_keys = {s["key"] for s in train_seqs}
    remaining = [s for s in seqs if s["key"] not in train_keys]
    test_seqs = pick_sequences(remaining, NUM_TEST_SEQS) if remaining else train_seqs[:1]

    # Clean old data
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 4: Build train split ({NUM_TRAIN_SEQS} seqs x {FRAMES_PER_SEQ} frames) ===")
    train_data, train_pcl = build_split("train", train_seqs)

    print(f"\n=== Step 5: Build test split ({NUM_TEST_SEQS} seqs x {FRAMES_PER_SEQ} frames) ===")
    test_data, test_pcl = build_split("test", test_seqs)

    print("\n=== Step 6: Write metadata pickles ===")
    with gzip.open(DATA_DIR / "train.pickle.gz", "wb") as f:
        pickle.dump(train_data, f)
    with gzip.open(DATA_DIR / "test.pickle.gz", "wb") as f:
        pickle.dump(test_data, f)
    valid = {k: list(range(len(v["timestamps"]))) for k, v in train_data.items()}
    with gzip.open(DATA_DIR / "valid_seq_ids.train.pickle.gz", "wb") as f:
        pickle.dump(valid, f)

    print("\n=== Step 7: Write COLMAP tars ===")
    write_pcl_tar(train_pcl, "train", DATA_DIR / "pcl.train.tar")
    write_pcl_tar(test_pcl, "test", DATA_DIR / "pcl.test.tar")

    print("\n=== Step 8: Write split files ===")
    split_lines = []
    for k, d in test_data.items():
        n = len(d["timestamps"])
        split_lines.append(f"{k} 0 {min(1, n-1)} {min(2, n-1)} {min(3, n-1)}")
    content = "\n".join(split_lines) + "\n"
    split_dir = PROJECT_ROOT / "splits" / "re10k_mine_filtered"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("test_files.txt", "val_files.txt"):
        (split_dir / name).write_text(content, encoding="utf-8")

    print(f"\n=== Summary ===")
    total_train = sum(len(d["timestamps"]) for d in train_data.values())
    total_test = sum(len(d["timestamps"]) for d in test_data.values())
    print(f"  train: {len(train_data)} seqs, {total_train} frames")
    print(f"  test:  {len(test_data)} seqs, {total_test} frames")
    print(f"  data dir: {DATA_DIR}")
    print(f"\nReady for 200-epoch CPU training.")


if __name__ == "__main__":
    main()
