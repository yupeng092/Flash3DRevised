from PIL import Image
import numpy as np
import os

print("=" * 72)
print("flash3d_native_multiview.py 原始渲染 vs RealESRGAN 超分 对比")
print("=" * 72)

orig_dir = "outputs/flash3d_native_v3/rgb"
sr_dir = "outputs/flash3d_native_v3_sr"

files = sorted(os.listdir(orig_dir))

print(f"\n{'视角':<14} {'对应原图':>16} {'原始分辨率':>12} {'超分分辨率':>12} {'原始大小':>10} {'超分大小':>10}")
print("-" * 80)

# 视角对应关系
pose_names = ["DSC_0286", "DSC_0291", "DSC_0287", "DSC_0293", "DSC_0292",
              "DSC_0295", "DSC_0289", "DSC_0290", "DSC_0288", "DSC_0294"]
angles = [0.0, 1.4, 2.7, 3.0, 3.5, 4.4, 4.6, 5.0, 5.0, 6.1]

orig_total = 0
sr_total = 0
for i, f in enumerate(files):
    o_im = Image.open(os.path.join(orig_dir, f))
    s_im = Image.open(os.path.join(sr_dir, f))
    o_kb = os.path.getsize(os.path.join(orig_dir, f)) // 1024
    s_kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024
    orig_total += o_kb
    sr_total += s_kb
    print(f"{f:<14} {pose_names[i]:>16} {o_im.size[0]}x{o_im.size[1]:>5} {s_im.size[0]}x{s_im.size[1]:>5} {o_kb:>8}KB {s_kb:>8}KB")

print("-" * 80)
print(f"{'总计':<14} {'':>16} {'':>12} {'':>12} {orig_total:>8}KB {sr_total:>8}KB")

# 画质指标
print(f"\n{'=' * 72}")
print("画质指标对比")
print(f"{'=' * 72}")

print(f"\n{'视角':<14} {'角度':>6} {'信息熵(原始)':>14} {'信息熵(超分)':>14} {'提升':>8} {'均值':>8} {'标准差':>8}")
print("-" * 72)

def entropy(path):
    arr = np.array(Image.open(path).convert("L"))
    hist = np.histogram(arr, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

entropy_orig = []
entropy_sr = []
for i, f in enumerate(files):
    e_o = entropy(os.path.join(orig_dir, f))
    e_s = entropy(os.path.join(sr_dir, f))
    entropy_orig.append(e_o)
    entropy_sr.append(e_s)

    # 计算像素分布
    o_arr = np.array(Image.open(os.path.join(orig_dir, f)).convert("L")).astype(float)
    s_arr = np.array(Image.open(os.path.join(sr_dir, f)).convert("L")).astype(float)

    print(f"{f:<14} {angles[i]:>5.1f}° {e_o:>14.2f} {e_s:>14.2f} {e_s-e_o:>+8.2f} {o_arr.mean():>8.1f} {o_arr.std():>8.1f}")

print(f"\n{'平均':>20} {np.mean(entropy_orig):>14.2f} {np.mean(entropy_sr):>14.2f} {np.mean(entropy_sr)-np.mean(entropy_orig):>+8.2f}")

# 视角间差异验证
print(f"\n{'=' * 72}")
print("3D 视差验证 (视角间平均像素差异)")
print(f"{'=' * 72}")

orig_imgs = [np.array(Image.open(os.path.join(orig_dir, f)).convert("RGB")).astype(float) for f in files]
sr_imgs = [np.array(Image.open(os.path.join(sr_dir, f)).convert("RGB")).astype(float) for f in files]

# 计算所有视角对的平均差异
orig_diffs = []
sr_diffs = []
for i in range(len(files)):
    for j in range(i+1, len(files)):
        d_o = np.abs(orig_imgs[i] - orig_imgs[j]).mean()
        d_s = np.abs(sr_imgs[i] - sr_imgs[j]).mean()
        orig_diffs.append(d_o)
        sr_diffs.append(d_s)

print(f"  原始渲染视角间差异: 均值={np.mean(orig_diffs):.2f}, 范围=[{np.min(orig_diffs):.1f}, {np.max(orig_diffs):.1f}]")
print(f"  超分后视角间差异:   均值={np.mean(sr_diffs):.2f}, 范围=[{np.min(sr_diffs):.1f}, {np.max(sr_diffs):.1f}]")

print(f"\n{'=' * 72}")
print("总结")
print(f"{'=' * 72}")
print(f"""
渲染脚本: flash3d_native_multiview.py
  - UniDepth V1 ViT-L/14 深度预测 (NystromAttention 已修复)
  - Flash3D 高斯参数预测 (286,720 个高斯)
  - COLMAP 真实多机位位姿 (10 个相机, 0-6.1°)
  - RANSAC 尺度对齐 (3,423 个稀疏深度点)
  - render_predicted_torch (3DGS α-blending)

超分: RealESRGAN x4 (RRDBNet, 1670万参数)

原始渲染: outputs/flash3d_native_v3/rgb/       (384x256, {orig_total/1024:.1f}MB)
超分结果: outputs/flash3d_native_v3_sr/        (1536x1024, {sr_total/1024:.1f}MB)

分辨率提升: 16x
文件大小提升: {sr_total/orig_total:.1f}x
信息熵提升: +{np.mean(entropy_sr)-np.mean(entropy_orig):.2f} (平均)
""")
