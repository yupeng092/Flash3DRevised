from PIL import Image
import numpy as np
import json
import os

print("=== COLMAP scale=2.0 render quality ===")
report_path = "outputs/courtyard_colmap_s2/rendered/render_report.json"
with open(report_path) as f:
    report = json.load(f)

indices = [0, 5, 10, 15, 20, 25, 30, 35]
rgb_dir = "outputs/courtyard_colmap_s2/rendered/rgb"
alpha_dir = "outputs/courtyard_colmap_s2/rendered/alpha"

n_files = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
print(f"Total rendered files: {n_files}")
print()

for i in range(n_files):
    r = np.array(Image.open(f"{rgb_dir}/view_{i:03d}.png").convert("RGB"))
    a = np.array(Image.open(f"{alpha_dir}/view_{i:03d}.png").convert("L"))
    gray_bg = np.all(np.abs(r - 128) < 15, axis=2).mean() * 100
    low_alpha = (a < 50).mean() * 100
    vis = report["visible_gaussians"][i]
    print(f"view_{i:03d} (pose {indices[i]:02d}): visible={vis:6d}, alpha={a.mean():.0f}, gray_bg={gray_bg:.1f}%, uncovered={low_alpha:.1f}%")

print()
print("=== Best views (sorted by coverage) ===")
results = []
for i in range(n_files):
    a = np.array(Image.open(f"{alpha_dir}/view_{i:03d}.png").convert("L"))
    results.append((i, indices[i], a.mean(), report["visible_gaussians"][i]))
results.sort(key=lambda x: -x[2])
for rank, (i, idx, amean, vis) in enumerate(results):
    print(f"  #{rank+1}: view_{i:03d} (pose {idx:02d}) alpha={amean:.0f} visible={vis}")
