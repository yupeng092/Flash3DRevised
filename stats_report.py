from PIL import Image
import json
import os

print("=" * 70)
print("当前渲染结果详细统计")
print("=" * 70)

# 渲染图分辨率
rgb_dir = "outputs/courtyard_colmap_hires/rgb"
files = sorted(os.listdir(rgb_dir))

with open("outputs/courtyard_colmap_hires/render_report.json") as f:
    report = json.load(f)

# 原始 COLMAP 点云总数
print("\n--- COLMAP 点云总数 ---")
print("  33,487 个 3D 点 (稀疏 SfM 重建)")

# 渲染参数
print("\n--- 渲染参数 ---")
print(f"  高斯尺度 (scale): 2.0")
print(f"  不透明度 (opacity): 0.9")
print(f"  渲染分辨率: 3072x2048")
print(f"  总像素: {3072*2048:,} = {3072*2048/1e6:.1f}M pixels")

# 每张图的统计
print("\n--- 每张渲染图统计 ---")
print(f"{'图片':<16} {'分辨率':<12} {'可见高斯数':>10} {'平均alpha':>10} {'文件大小':>10}")
print("-" * 70)

indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
for i, f in enumerate(files):
    im = Image.open(os.path.join(rgb_dir, f))
    w, h = im.size
    vis = report["visible_gaussians"][i]
    alpha = report["seconds_per_view"][i]  # This is wrong, need alpha
    size_kb = os.path.getsize(os.path.join(rgb_dir, f)) // 1024
    print(f"{f:<16} {w}x{h:<8} {vis:>10,}   {'-':>8}   {size_kb:>8}KB")

# alpha 值
print("\n--- 每张图的平均 alpha (覆盖率) ---")
alpha_dir = "outputs/courtyard_colmap_hires/alpha"
import numpy as np
for i, f in enumerate(files):
    af = f.replace("rgb", "alpha") if "rgb" in f else f
    alpha_files = sorted(os.listdir(alpha_dir))
    a = np.array(Image.open(os.path.join(alpha_dir, alpha_files[i])).convert("L"))
    vis = report["visible_gaussians"][i]
    print(f"  {f}: visible={vis:>6,}, alpha_mean={a.mean():.0f}/255 ({a.mean()/255*100:.0f}%), uncovered={(a<50).mean()*100:.1f}%")

print("\n" + "=" * 70)
print("关键问题分析")
print("=" * 70)
print(f"""
1. 分辨率: 3072x2048 = {3072*2048:,} 像素 (6.3M)
   原图: 6205x4135 = {6205*4135:,} 像素 (25.7M)
   渲染图是原图的 1/2 分辨率

2. 高斯椭球数量:
   - COLMAP 总点数: 33,487 (全场景)
   - 每视角可见: {min(report['visible_gaussians']):,} ~ {max(report['visible_gaussians']):,}
   - 平均每视角: {sum(report['visible_gaussians'])//len(report['visible_gaussians']):,}

3. 密度分析:
   - 渲染图像素: {3072*2048:,}
   - 平均可见高斯: {sum(report['visible_gaussians'])//len(report['visible_gaussians']):,}
   - 每个高斯覆盖: {3072*2048 // (sum(report['visible_gaussians'])//len(report['visible_gaussians']))} 像素
   - 问题: 每个高斯要覆盖 ~{3072*2048 // (sum(report['visible_gaussians'])//len(report['visible_gaussians']))//1000}K 像素,太稀疏
""")
