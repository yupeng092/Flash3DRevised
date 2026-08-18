from PIL import Image
import numpy as np
import os

print("=" * 70)
print("render_cpu_alpha 原始渲染 vs RealESRGAN 超分 对比")
print("=" * 70)

orig_dir = "outputs/flash3d_render_cpu/rendered/rgb"
sr_dir = "outputs/flash3d_render_cpu_sr"

files = sorted(os.listdir(orig_dir))

print(f"\n{'视角':<14} {'原始分辨率':>12} {'超分分辨率':>12} {'原始大小':>10} {'超分大小':>10} {'像素倍数':>8}")
print("-" * 78)

orig_sizes = []
sr_sizes = []
for f in files:
    o_im = Image.open(os.path.join(orig_dir, f))
    s_im = Image.open(os.path.join(sr_dir, f))
    o_kb = os.path.getsize(os.path.join(orig_dir, f)) // 1024
    s_kb = os.path.getsize(os.path.join(sr_dir, f)) // 1024
    o_px = o_im.size[0] * o_im.size[1]
    s_px = s_im.size[0] * s_im.size[1]
    ratio = s_px / o_px
    print(f"{f:<14} {o_im.size[0]}x{o_im.size[1]:>5} {s_im.size[0]}x{s_im.size[1]:>5} {o_kb:>8}KB {s_kb:>8}KB {ratio:>7.0f}x")
    orig_sizes.append(o_kb)
    sr_sizes.append(s_kb)

print(f"\n{'总计':<14} {'':>12} {'':>12} {sum(orig_sizes):>8}KB {sum(sr_sizes):>8}KB")

# 画质指标对比
print(f"\n{'=' * 70}")
print("画质指标: 信息熵 (越高=细节越丰富)")
print(f"{'=' * 70}")
print(f"\n{'视角':<14} {'原始(384x256)':>14} {'超分(1536x1024)':>18} {'提升':>8}")
print("-" * 58)

for f in files:
    def entropy(path):
        arr = np.array(Image.open(path).convert("L"))
        hist = np.histogram(arr, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    e_o = entropy(os.path.join(orig_dir, f))
    e_s = entropy(os.path.join(sr_dir, f))
    print(f"{f:<14} {e_o:>14.2f} {e_s:>18.2f} {e_s-e_o:>+8.2f}")

print(f"\n{'=' * 70}")
print("总结")
print(f"{'=' * 70}")
print(f"""
原始渲染: render_cpu_alpha.py 输出 (384x256, 共 {sum(orig_sizes)/1024:.1f}MB)
超分结果: RealESRGAN x4 (1536x1024, 共 {sum(sr_sizes)/1024:.1f}MB)

分辨率提升: 16x (像素数)
文件大小提升: {sum(sr_sizes)/sum(orig_sizes):.1f}x

文件位置:
  原始: outputs/flash3d_render_cpu/rendered/rgb/
  超分: outputs/flash3d_render_cpu_sr/
""")
