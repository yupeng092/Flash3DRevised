from PIL import Image
import numpy as np
import json

print("=== COLMAP render quality analysis ===")
with open("outputs/courtyard_colmap/rendered/render_report.json") as f:
    report = json.load(f)

indices = [0, 5, 10, 15, 20, 25, 30, 35]
for i, idx in enumerate(indices):
    r = np.array(Image.open(f"outputs/courtyard_colmap/rendered/rgb/view_{i:03d}.png").convert("RGB"))
    a = np.array(Image.open(f"outputs/courtyard_colmap/rendered/alpha/view_{i:03d}.png").convert("L"))
    gray_bg = np.all(np.abs(r - 128) < 15, axis=2).mean() * 100
    low_alpha = (a < 50).mean() * 100
    vis = report["visible_gaussians"][i]
    print(f"view_{i:03d} (pose {idx:02d}): visible={vis:6d}, alpha_mean={a.mean():.0f}, gray_bg={gray_bg:.1f}%, uncovered={low_alpha:.1f}%")
