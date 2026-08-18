from PIL import Image
import numpy as np
import os
import json

print("=" * 78)
print("benchmark_cpu 推理 → render_cpu_alpha 渲染 → 点云增强 → RealESRGAN 超分")
print("基线 vs 增强版 全流程对比")
print("=" * 78)

dirs = {
    "基线原始": "outputs/benchmark_render_baseline/rgb",
    "基线超分": "outputs/benchmark_baseline_sr",
    "增强原始": "outputs/benchmark_render_enhanced/rgb",
    "增强超分": "outputs/benchmark_enhanced_sr",
}

pose_names = ["DSC_0286", "DSC_0291", "DSC_0287", "DSC_0293", "DSC_0292",
              "DSC_0295", "DSC_0289", "DSC_0290", "DSC_0288", "DSC_0294"]
angles = [0.0, 1.4, 2.7, 3.0, 3.5, 4.4, 4.6, 5.0, 5.0, 6.1]

# Alpha coverage from render reports
with open("outputs/benchmark_render_baseline/render_report.json") as f:
    baseline_report = json.load(f)
with open("outputs/benchmark_render_enhanced/render_report.json") as f:
    enhanced_report = json.load(f)

def entropy(path):
    arr = np.array(Image.open(path).convert("L"))
    hist = np.histogram(arr, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

print(f"\n{'视角':<10} {'对应图':>8} {'角度':>5} │ {'基线alpha':>9} {'增强alpha':>9} │ {'基线可见':>8} {'增强可见':>8} │ {'基线原始':>8} {'基线超分':>8} {'增强原始':>8} {'增强超分':>8}")
print("─" * 120)

for i in range(10):
    f = f"view_{i:03d}.png"
    b_alpha = baseline_report["seconds_per_view"][i]  # not alpha, use alpha dir
    e_alpha = enhanced_report["seconds_per_view"][i]

    # Read alpha from alpha images
    b_alpha_img = np.array(Image.open(f"outputs/benchmark_render_baseline/alpha/{f}").convert("L"))
    e_alpha_img = np.array(Image.open(f"outputs/benchmark_render_enhanced/alpha/{f}").convert("L"))
    b_alpha_mean = b_alpha_img.mean() / 255
    e_alpha_mean = e_alpha_img.mean() / 255

    b_vis = baseline_report["visible_gaussians"][i]
    e_vis = enhanced_report["visible_gaussians"][i]

    b_orig_kb = os.path.getsize(os.path.join(dirs["基线原始"], f)) // 1024
    b_sr_kb = os.path.getsize(os.path.join(dirs["基线超分"], f)) // 1024
    e_orig_kb = os.path.getsize(os.path.join(dirs["增强原始"], f)) // 1024
    e_sr_kb = os.path.getsize(os.path.join(dirs["增强超分"], f)) // 1024

    print(f"view_{i:03d}   {pose_names[i]:>8} {angles[i]:>4.1f}° │ {b_alpha_mean:>9.2f} {e_alpha_mean:>9.2f} │ {b_vis:>8,} {e_vis:>8,} │ {b_orig_kb:>7}KB {b_sr_kb:>7}KB {e_orig_kb:>7}KB {e_sr_kb:>7}KB")

# Totals
print("─" * 120)
totals = {}
for label, d in dirs.items():
    totals[label] = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".png")) // 1024
print(f"{'总计':>26} │ {'':>9} {'':>9} │ {'':>8} {'':>8} │ {totals['基线原始']:>7}KB {totals['基线超分']:>7}KB {totals['增强原始']:>7}KB {totals['增强超分']:>7}KB")

# Entropy comparison
print(f"\n{'=' * 78}")
print("信息熵对比 (越高=细节越丰富)")
print(f"{'=' * 78}")
print(f"\n{'视角':<10} {'基线原始':>10} {'基线超分':>10} {'提升':>8} │ {'增强原始':>10} {'增强超分':>10} {'提升':>8} │ {'增强-基线':>10}")
print("─" * 88)

entropies = {}
for label, d in dirs.items():
    entropies[label] = []
    for i in range(10):
        f = f"view_{i:03d}.png"
        entropies[label].append(entropy(os.path.join(d, f)))

for i in range(10):
    f = f"view_{i:03d}"
    b_o = entropies["基线原始"][i]
    b_s = entropies["基线超分"][i]
    e_o = entropies["增强原始"][i]
    e_s = entropies["增强超分"][i]
    print(f"{f:<10} {b_o:>10.2f} {b_s:>10.2f} {b_s-b_o:>+8.2f} │ {e_o:>10.2f} {e_s:>10.2f} {e_s-e_o:>+8.2f} │ {e_s-b_s:>+10.2f}")

print("─" * 88)
print(f"{'平均':<10} {np.mean(entropies['基线原始']):>10.2f} {np.mean(entropies['基线超分']):>10.2f} {np.mean(entropies['基线超分'])-np.mean(entropies['基线原始']):>+8.2f} │ {np.mean(entropies['增强原始']):>10.2f} {np.mean(entropies['增强超分']):>10.2f} {np.mean(entropies['增强超分'])-np.mean(entropies['增强原始']):>+8.2f} │ {np.mean(entropies['增强超分'])-np.mean(entropies['基线超分']):>+10.2f}")

print(f"\n{'=' * 78}")
print("总结")
print(f"{'=' * 78}")
print(f"""
流程:
  1. benchmark_cpu.py 推理 → 286,720 个高斯 (Flash3D + UniDepth V1)
  2. render_cpu_alpha.py 渲染 → 10 张 384×256 (基线)
  3. 点云增强: 2x 尺度 + 1.5x opacity + 2x 密度 → 573,440 个高斯
  4. render_cpu_alpha.py 重新渲染 → 10 张 384×256 (增强)
  5. RealESRGAN x4 超分 → 10 张 1536×1024 (基线超分 + 增强超分)

点云增强效果:
  - 可见高斯数: 基线 {sum(baseline_report['visible_gaussians'])//10:,} → 增强 {sum(enhanced_report['visible_gaussians'])//10:,} (平均)
  - Alpha 覆盖: 基线 {np.mean([np.array(Image.open(f'outputs/benchmark_render_baseline/alpha/view_{i:03d}.png').convert('L')).mean()/255 for i in range(10)]):.2f} → 增强 {np.mean([np.array(Image.open(f'outputs/benchmark_render_enhanced/alpha/view_{i:03d}.png').convert('L')).mean()/255 for i in range(10)]):.2f}

超分效果:
  - 基线: 信息熵 {np.mean(entropies['基线原始']):.2f} → {np.mean(entropies['基线超分']):.2f} (+{np.mean(entropies['基线超分'])-np.mean(entropies['基线原始']):.2f})
  - 增强: 信息熵 {np.mean(entropies['增强原始']):.2f} → {np.mean(entropies['增强超分']):.2f} (+{np.mean(entropies['增强超分'])-np.mean(entropies['增强原始']):.2f})

输出位置:
  基线原始: outputs/benchmark_render_baseline/rgb/     (384×256)
  基线超分: outputs/benchmark_baseline_sr/              (1536×1024)
  增强原始: outputs/benchmark_render_enhanced/rgb/     (384×256)
  增强超分: outputs/benchmark_enhanced_sr/              (1536×1024)
""")
