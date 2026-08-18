import torch
import numpy as np
import math

# Load gaussians
data = torch.load("outputs/benchmark_enhanced/gaussians.pt", map_location="cpu", weights_only=False)
g = data["gaussians"]

# Simulate the projection to find the right scale_modifier
xyz = g["xyz"]
scaling = g["scaling"]
z = xyz[:, 2]
fx = 211.0  # render focal length

# 3D scaling -> 2D radius approximation
# covariance_2d ≈ (fx * scale / z)^2  (simplified)
# radius = sigma_cutoff * sqrt(covariance_2d) = sigma_cutoff * fx * scale / z
# We want radius ≈ 2-3 pixels for good coverage
# sigma_cutoff = 3.0 (default)

sigma_cutoff = 3.0
z_median = z.median().item()
scale_median = scaling.median().item()

# Current radius with scale_modifier=1.0
current_radius = sigma_cutoff * fx * scale_median / z_median
print(f"z median: {z_median:.2f}")
print(f"scaling median: {scale_median:.6f}")
print(f"fx: {fx}")
print(f"sigma_cutoff: {sigma_cutoff}")
print(f"Current 2D radius (scale_modifier=1.0): {current_radius:.4f} pixels")
print()

# We want radius ≈ 3 pixels
target_radius = 3.0
needed_scale_modifier = target_radius / (sigma_cutoff * fx * scale_median / z_median)
print(f"Target radius: {target_radius} pixels")
print(f"Needed scale_modifier: {needed_scale_modifier:.1f}")
print()

# Also check with different targets
for target in [1, 2, 3, 5, 8, 10]:
    sm = target / (sigma_cutoff * fx * scale_median / z_median)
    print(f"  target_radius={target}px -> scale_modifier={sm:.1f}")

# Also check the Flash3D native renderer's approach
# In npu_differentiable_renderer.py, scaling is used directly (no scale_modifier)
# But the covariance projection includes the jacobian with fx/z
# So the 2D size = sigma_cutoff * sqrt(eigenvalue) where eigenvalue ~ (fx*scale/z)^2
# This is the same formula

print()
print("=== render_cpu_multiview.py 默认 scale_modifier=0.55 ===")
print(f"  实际 radius: {sigma_cutoff * fx * scale_median * 0.55 / z_median:.4f} pixels")
print(f"  这就是为什么画面模糊: 每个高斯只有 {sigma_cutoff * fx * scale_median * 0.55 / z_median:.2f} 像素")
