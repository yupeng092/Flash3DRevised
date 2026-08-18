from PIL import Image
import numpy as np
import os
import json

print("=" * 78)
print("render_cpu_multiview.py 渲染 vs RealESRGAN 超分 对比")
print("=" * 78)

orig_dir = "outputs/benchmark_multiview_render/rgb"
sr_dir = "outputs/benchmark_multiview_sr"
alpha_dir = "outputs/benchmark_multiview_render/alpha"

with open("outputs/benchmark_multiview_render/multiview_report.json") as f:
    report = json.load(f)

files = sorted(os.listdir(orig_dir))

print(f"\n{'视角':<20} {'可见高斯':>8} {'alpha':>6} {'原始分辨率':>12} {'超分分辨率':>12} {'原始大小':>10} {'超分大小':>10}")
print("-" * 90)

orig_total = 0
sr_total = 0
for i, f in enumerate(files):
    o_im = Image.open(os.path.join(orig_dir, f))
    s_im = Image.open(os.path.join(sr_dir, f))
    o_kb = os.path.getsize(os.path.join(orig_dir, f)) // 1024
    s_kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024
    orig_total += o_kb
    sr_total += s_kb
    
    cam = report["cameras"][i]
    vis = cam["visible_gaussians"]
    alpha = cam["mean_alpha"]
    
    print(f"{f:<20} {vis:>8,} {alpha:>6.2f} {o_im.size[0]}x{o_im.size[1]:>5} {s_im.size[0]}x{s_im.size[1]:>5} {o_kb:>8}KB {s_kb:>8}KB")

print("-" * 90)
print(f"{'总计':<20} {'':>8} {'':>6} {'':>12} {'':>12} {orig_total:>8}KB {sr_total:>8}KB")

# Entropy comparison
print(f"\n{'=' * 78}")
print("信息熵对比 (越高=细节越丰富)")
print(f"{'=' * 78}")
print(f"\n{'视角':<20} {'信息熵(原始)':>14} {'信息熵(超分)':>14} {'提升':>8}")
print("-" * 60)

def entropy(path):
    arr = np.array(Image.open(path).convert("L"))
    hist = np.histogram(arr, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

entropies_o = []
entropies_s = []
for f in files:
    e_o = entropy(os.path.join(orig_dir, f))
    e_s = entropy(os.path.join(sr_dir, f))
    entropies_o.append(e_o)
    entropies_s.append(e_s)
    print(f"{f:<20} {e_o:>14.2f} {e_s:>14.2f} {e_s-e_o:>+8.2f}")

print(f"\n{'平均':<20} {np.mean(entropies_o):>14.2f} {np.mean(entropies_s):>14.2f} {np.mean(entropies_s)-np.mean(entropies_o):>+8.2f}")

print(f"\n{'=' * 78}")
print("总结")
print(f"{'=' * 78}")
print(f"""
渲染脚本: render_cpu_multiview.py
  - 读取 benchmark_cpu 推理的增强点云 (573,440 高斯)
  - 使用 COLMAP 10 个真实多机位 (camera_rig.json)
  - 3DGS α-blending 渲染

超分: RealESRGAN x4 (RRDBNet, 1670万参数)

原始渲染: outputs/benchmark_multiview_render/rgb/  (384x256, {orig_total/1024:.1f}MB)
超分结果: outputs/benchmark_multiview_sr/           (1536x1024, {sr_total/1024:.1f}MB)

分辨率提升: 16x
文件大小提升: {sr_total/orig_total:.1f}x
信息熵提升: +{np.mean(entropies_s)-np.mean(entropies_o):.2f}
""")
