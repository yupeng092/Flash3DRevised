from PIL import Image
import numpy as np
import os

print("=" * 70)
print("RealESRGAN 超分结果对比")
print("=" * 70)

# 三个版本对比
dirs = {
    "低分辨率(384x256)": "outputs/courtyard_colmap_final/rgb",
    "高分辨率(3072x2048, 无超分)": "outputs/courtyard_colmap_hires/rgb",
    "超分(768x512 -> 3072x2048, RealESRGAN x4)": "outputs/courtyard_colmap_sr",
}

for label, d in dirs.items():
    if not os.path.exists(d):
        continue
    files = sorted(os.listdir(d))
    total = sum(os.path.getsize(os.path.join(d, f)) for f in files if f.endswith(".png"))
    sizes = []
    for f in files:
        if f.endswith(".png"):
            im = Image.open(os.path.join(d, f))
            sizes.append(im.size)
    print(f"\n{label}:")
    print(f"  图片数: {len(sizes)}")
    if sizes:
        print(f"  分辨率: {sizes[0][0]}x{sizes[0][1]}")
        print(f"  总大小: {total/1024/1024:.1f} MB")
        print(f"  平均每张: {total/len(sizes)/1024:.0f} KB")

# 详细对比每张超分图
print("\n" + "=" * 70)
print("超分图片详细信息")
print("=" * 70)
sr_dir = "outputs/courtyard_colmap_sr"
files = sorted(os.listdir(sr_dir))
print(f"\n{'图片':<16} {'分辨率':<14} {'文件大小':>10}")
print("-" * 44)
for f in files:
    im = Image.open(os.path.join(sr_dir, f))
    kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024
    print(f"{f:<16} {im.size[0]}x{im.size[1]:<8} {kb:>8} KB")

# 画质对比: 超分前 vs 超分后 的信息熵(细节丰富度)
print("\n" + "=" * 70)
print("画质指标: 图像信息熵(越高=细节越丰富)")
print("=" * 70)

midres_dir = "outputs/courtyard_colmap_midres/rgb"
hires_dir = "outputs/courtyard_colmap_hires/rgb"

print(f"\n{'图片':<16} {'768x512(原始)':>14} {'3072x2048(直接)':>16} {'3072x2048(超分)':>16}")
print("-" * 66)

mid_files = sorted(os.listdir(midres_dir))
hi_files = sorted(os.listdir(hires_dir))
sr_files = sorted(os.listdir(sr_dir))

for f in sr_files:
    # 信息熵计算
    def entropy(im_path):
        arr = np.array(Image.open(im_path).convert("L"))
        hist = np.histogram(arr, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    mid_path = os.path.join(midres_dir, f.replace("view_0", "view_0"))
    hi_path = os.path.join(hires_dir, f)
    sr_path = os.path.join(sr_dir, f)

    e_mid = entropy(mid_path) if os.path.exists(mid_path) else 0
    e_hi = entropy(hi_path) if os.path.exists(hi_path) else 0
    e_sr = entropy(sr_path)

    print(f"{f:<16} {e_mid:>14.2f} {e_hi:>16.2f} {e_sr:>16.2f}")
