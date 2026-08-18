#!/usr/bin/env python3
"""Enhance benchmark_cpu gaussians.pt: increase splat scale and opacity.

Creates an enhanced version of the point cloud by:
1. Increasing gaussian scaling (larger splats -> better coverage)
2. Increasing opacity (more opaque -> less see-through gaps)
3. Duplicating points with slight offsets (higher density)

This produces gaussians_enhanced.pt for render_cpu_alpha.py.
"""
import torch
import numpy as np
from pathlib import Path

def main():
    input_path = Path("outputs/benchmark_scaled/gaussians.pt")
    output_path = Path("outputs/benchmark_enhanced/gaussians.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark gaussians...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    g = data["gaussians"]
    n = g["xyz"].shape[0]
    print(f"  Original: {n} gaussians")
    print(f"  Scaling range: [{g['scaling'].min():.4f}, {g['scaling'].max():.4f}]")
    print(f"  Opacity range: [{g['opacity'].min():.4f}, {g['opacity'].max():.4f}]")

    # Enhancement 1: Increase splat scale by 2x
    scale_boost = 2.0
    g_enhanced = {k: v.clone() for k, v in g.items()}
    g_enhanced["scaling"] = g["scaling"] * scale_boost

    # Enhancement 2: Boost opacity (push towards 1.0)
    g_enhanced["opacity"] = (g["opacity"] * 1.5).clamp(0, 1.0)

    # Enhancement 3: Duplicate points with small random offsets for density
    np.random.seed(42)
    n_dup = n  # duplicate all
    offset = torch.from_numpy(np.random.randn(n_dup, 3).astype(np.float32) * 0.01)  # small offset

    g_dense = {}
    for k in g_enhanced:
        orig = g_enhanced[k]
        if k == "xyz":
            dup = orig + offset
            g_dense[k] = torch.cat([orig, dup], dim=0)
        elif k == "opacity":
            dup = orig * 0.8  # duplicated points slightly more transparent
            g_dense[k] = torch.cat([orig, dup], dim=0)
        elif k == "scaling":
            dup = orig * 0.7  # duplicated points smaller
            g_dense[k] = torch.cat([orig, dup], dim=0)
        elif k == "rotation":
            g_dense[k] = torch.cat([orig, orig], dim=0)  # same rotation
        elif k == "features_dc":
            g_dense[k] = torch.cat([orig, orig], dim=0)  # same color
        elif k == "features_rest":
            g_dense[k] = torch.cat([orig, orig], dim=0)
        else:
            g_dense[k] = torch.cat([orig, orig], dim=0)

    n_final = g_dense["xyz"].shape[0]
    print(f"\nEnhanced: {n_final} gaussians (2x density)")
    print(f"  Scaling boost: {scale_boost}x")
    print(f"  Opacity boost: 1.5x (clamped to 1.0)")
    print(f"  Scaling range: [{g_dense['scaling'].min():.4f}, {g_dense['scaling'].max():.4f}]")
    print(f"  Opacity range: [{g_dense['opacity'].min():.4f}, {g_dense['opacity'].max():.4f}]")

    metadata = data.get("metadata", {})
    metadata["enhanced"] = True
    metadata["scale_boost"] = scale_boost
    metadata["doubled_density"] = True
    torch.save({"gaussians": g_dense, "metadata": metadata}, output_path)
    print(f"\nSaved to {output_path}")

    # Copy poses
    import shutil
    poses_src = Path("outputs/benchmark_scaled/render_poses.pt")
    poses_dst = Path("outputs/benchmark_enhanced/render_poses.pt")
    shutil.copy(poses_src, poses_dst)
    print(f"Copied poses to {poses_dst}")

if __name__ == "__main__":
    main()
