import numpy as np
import json

# Read all 38 COLMAP images
images = []
with open(r"D:\Python Project\courtyard\dslr_calibration_undistorted\images.txt") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        images.append({
            "id": int(parts[0]),
            "qvec": [float(x) for x in parts[1:5]],
            "tvec": [float(x) for x in parts[5:8]],
            "name": parts[9],
        })
        i += 1

def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)

# Find ref (DSC_0286)
ref = [img for img in images if "DSC_0286" in img["name"]][0]
R0 = quat_to_rot(ref["qvec"])
t0 = np.array(ref["tvec"])

# Compute relative pose for all 38
results = []
for img in images:
    Ri = quat_to_rot(img["qvec"])
    ti = np.array(img["tvec"])
    R_rel = Ri @ R0.T
    t_rel = ti - R_rel @ t0
    trans = np.linalg.norm(t_rel)
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    results.append({"name": img["name"], "angle": angle, "trans": trans, "idx": images.index(img)})

# Sort by angle
results.sort(key=lambda x: x["angle"])

print("=== 全部 38 个相机 (按角度排序) ===")
print(f"{'序号':>4} {'图片名':>28} {'角度':>8} {'平移':>8}")
print("-" * 55)
for r in results:
    print(f"{r['idx']:>4} {r['name']:>28} {r['angle']:>8.1f} {r['trans']:>8.2f}")

# Select 10 with smallest angle (excluding 0° which is the reference itself)
# Include the reference as view 0, then 9 others with smallest angles
non_ref = [r for r in results if r["angle"] > 0.5]
selected = [results[0]] + non_ref[:9]  # ref + 9 smallest

print(f"\n=== 选中的 10 个视角 ===")
print(f"{'序号':>4} {'图片名':>28} {'角度':>8} {'平移':>8}")
print("-" * 55)
for r in selected:
    print(f"{r['idx']:>4} {r['name']:>28} {r['angle']:>8.1f} {r['trans']:>8.2f}")

# Output as JSON for the render script
output = [{"index": r["idx"], "name": r["name"], "angle": r["angle"], "trans": r["trans"]} for r in selected]
with open("outputs/courtyard_flash3d_multiview/selected_poses.json", "w") as f:
    json.dump(output, f, indent=2)

# Print pose indices for render command
indices = [r["idx"] for r in selected]
print(f"\nPose indices: {indices}")
print(f"\nCommand:")
print(f"python render_flash3d_multiview.py --gaussians outputs/courtyard_benchmark/gaussians.pt \\")
print(f"  --colmap-dir \"D:\\Python Project\\courtyard\\dslr_calibration_undistorted\" \\")
print(f"  --output outputs/courtyard_flash3d_multiview_v2 \\")
print(f"  --ref-image DSC_0286 --render-height 256 --render-width 384 \\")
print(f"  --keep-ratio 0.35 --pose-indices {' '.join(str(i) for i in indices)}")
