from PIL import Image
import numpy as np
import os

print("=" * 70)
print("Flash3D 单图点云 → RealESRGAN 超分结果")
print("=" * 70)

orig_dir = "outputs/courtyard_render/rgb"
sr_dir = "outputs/courtyard_colmap_sr"

print("\n--- 超分前后对比 ---")
print(f"{'图片':<16} {'原始(256x384)':>16} {'超分(1024x1536)':>18} {'放大倍数':>10}")
print("-" * 66)

orig_files = sorted(os.listdir(orig_dir))
sr_files = sorted(os.listdir(sr_dir))[:10]  # 取前 10 个对应

for f in orig_files:
    im_o = Image.open(os.path.join(orig_dir, f))
    im_s = Image.open(os.path.join(sr_dir, f)) if os.path.exists(os.path.join(sr_dir, f)) else None

    o_kb = os.path.getsize(os.path.join(orig_dir, f)) // 1024
    s_kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024 if im_s else 0
    s_size = f"{im_s.size[0]}x{im_s.size[1]}" if im_s else "N/A"

    print(f"{f:<16} {im_o.size[0]}x{im_o.size[1]} ({o_kb}KB){'':>4} {s_size} ({s_kb}KB){'':>6} 4x")

print("\n--- 画质指标 ---")
print(f"{'图片':<16} {'信息熵(原始)':>14} {'信息熵(超分)':>14} {'提升':>8}")
print("-" * 56)

for f in orig_files:
    def entropy(path):
        arr = np.array(Image.open(path).convert("L"))
        hist = np.histogram(arr, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    e_o = entropy(os.path.join(orig_dir, f))
    e_s = entropy(os.path.join(sr_dir, f))
    print(f"{f:<16} {e_o:>14.2f} {e_s:>14.2f} {e_s-e_o:>+8.2f}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
输入: Flash3D 单图推理 (256x384, 286,720 个高斯)
处理: RealESRGAN x4 (RRDBNet, 1670万参数, 分块超分)
输出: 10 张 1024x1536 超分图

文件位置: outputs/courtyard_flash3d_sr/
""")
