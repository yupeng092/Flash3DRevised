from PIL import Image
import numpy as np
import json

files = ["view_000","view_005","view_010","view_015","view_020","view_025","view_030","view_035"]
with open("outputs/courtyard_colmap_s2/rendered/render_report.json") as f:
    report = json.load(f)

print("=== COLMAP scale=2.0 render quality ===")
for i, fn in enumerate(files):
    r = np.array(Image.open(f"outputs/courtyard_colmap_s2/rendered/rgb/{fn}.png").convert("RGB"))
    a = np.array(Image.open(f"outputs/courtyard_colmap_s2/rendered/alpha/{fn}.png").convert("L"))
    vis = report["visible_gaussians"][i]
    gray_bg = np.all(np.abs(r - 128) < 15, axis=2).mean() * 100
    uncovered = (a < 50).mean() * 100
    print(f"{fn}: visible={vis:6d}, alpha={a.mean():.0f}, gray_bg={gray_bg:.1f}%, uncovered={uncovered:.1f}%")
