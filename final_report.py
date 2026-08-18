from PIL import Image
import numpy as np
import os

print("=" * 70)
print("Flash3D 单图点云 + COLMAP 真实多机位 + RealESRGAN 超分")
print("=" * 70)

sr_dir = "outputs/courtyard_flash3d_multiview_sr"
render_dir = "outputs/courtyard_flash3d_multiview_v2/rendered/rgb"

import json
with open("outputs/courtyard_flash3d_multiview_v2/render_poses.json") as f:
    poses = json.load(f)

print(f"\n{'视角':<8} {'对应原图':>16} {'平移':>6} {'旋转':>6} {'可见高斯':>8} {'alpha':>6} {'超分后':>12} {'大小':>8}")
print("-" * 80)

files = sorted(os.listdir(sr_dir))
for i, f in enumerate(files):
    sr_im = Image.open(os.path.join(sr_dir, f))
    render_im = Image.open(os.path.join(render_dir, f))
    
    import json
    with open("outputs/courtyard_flash3d_multiview_v2/rendered/render_report.json") as rf:
        report = json.load(rf)
    
    p = poses[i]
    vis = report["visible_gaussians"][i]
    alpha = report["seconds_per_view"][i]  # not alpha, need to get from report
    
    # Get alpha from alpha image
    alpha_im = np.array(Image.open(f"outputs/courtyard_flash3d_multiview_v2/rendered/alpha/{f}").convert("L"))
    
    kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024
    name = p["name"].split("/")[-1]
    print(f"{f:<8} {name:>16} {p['trans']:>6.2f} {p['angle']:>6.1f} {vis:>8,} {alpha_im.mean()/255:>6.2f} {sr_im.size[0]}x{sr_im.size[1]:>6} {kb:>6}KB")

print(f"\n=== 视角间差异验证 (3D 视差) ===")
images_arr = []
for f in files:
    images_arr.append(np.array(Image.open(os.path.join(sr_dir, f)).convert("RGB")))

print(f"\n{'':>10}", end="")
for j in range(min(5, len(files))):
    print(f"  view_{j:03d}", end="")
print()
for i in range(min(5, len(files))):
    print(f"  view_{i:03d}", end="")
    for j in range(min(5, len(files))):
        if i == j:
            print(f"      0", end="")
        else:
            diff = np.abs(images_arr[i].astype(float) - images_arr[j].astype(float)).mean()
            print(f"   {diff:5.1f}", end="")
    print()

print(f"\n=== 输出位置 ===")
print(f"outputs/courtyard_flash3d_multiview_sr/")
print(f"  ├── view_000.png ~ view_009.png  (10 张 1536x1024 超分图)")
print(f"  总大小: {sum(os.path.getsize(os.path.join(sr_dir, f)) for f in files) / 1024 / 1024:.1f} MB")
