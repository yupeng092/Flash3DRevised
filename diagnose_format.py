"""Diagnose: benchmark_cpu output vs render_cpu_alpha/multiview expected input."""
import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("格式兼容性诊断")
print("=" * 70)

# 1. benchmark_cpu 原始输出
print("\n--- benchmark_cpu 原始输出 ---")
data = torch.load("outputs/benchmark_inference/gaussians.pt", map_location="cpu", weights_only=False)
g = data["gaussians"]
print(f"Keys: {list(g.keys())}")
for k, v in g.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")

print(f"\nMetadata: {data.get('metadata', {})}")

# 2. render_cpu_alpha 的 load_gaussians 期望格式
print("\n--- render_cpu_alpha load_gaussians 期望 ---")
print("  required: xyz, opacity, scaling, rotation, features_dc")
print("  opacity: reshape(-1).clamp(0,1)")
print("  color = (0.5 + SH_C0 * features_dc).clamp(0,1)  # degree-0 SH -> RGB")
print("  scaling: 直接用 (不经过 exp/activation)")
print("  rotation: 直接用 (不经过 normalize)")

# 3. Flash3D 原始输出的 scaling/rotation 格式
print("\n--- Flash3D 原始格式分析 ---")
print("  Flash3D 的 gauss_scaling 是经过 exp 激活的 (gaussian_decoder.py:97)")
print("  Flash3D 的 gauss_rotation 是经过 normalize 的 (gaussian_decoder.py:98)")
print("  Flash3D 的 gauss_opacity 是经过 sigmoid 的 (gaussian_decoder.py:96)")
print("  Flash3D 的 features_dc 是原始 SH DC 系数 (未加 0.5)")

# 4. 关键问题: scaling
print("\n--- scaling 问题分析 ---")
scaling = g["scaling"]
print(f"  benchmark_cpu scaling: range=[{scaling.min():.6f}, {scaling.max():.6f}]")
print(f"  mean={scaling.mean():.6f}, median={scaling.median():.6f}")
print(f"  这些值是 exp(scale_scale * raw + log(scale_bias)) 后的结果")
print(f"  render_cpu_alpha 直接用这个值作为高斯半径(像素)")

# 检查 scale_modifier 的影响
print(f"\n  render_cpu_alpha 默认 --scale-modifier 1.0")
print(f"  render_cpu_multiview 默认 --scale-modifier 0.55")
print(f"  实际使用的 scaling = benchmark_scaling * scale_modifier")
print(f"  如果 benchmark scaling 已经很小,再乘 0.55 会更小")

# 5. 模拟 render_cpu_alpha 的处理
print("\n--- 模拟 render_cpu_alpha 处理 ---")
from render_cpu_alpha import SH_C0, load_gaussians
gaussians_loaded, metadata = load_gaussians(
    Path("outputs/benchmark_enhanced/gaussians.pt"),
    keep_ratio=0.35, min_opacity=0.01, crop_padding=True
)
print(f"  加载后高斯数: {gaussians_loaded['xyz'].shape[0]}")
print(f"  opacity: range=[{gaussians_loaded['opacity'].min():.4f}, {gaussians_loaded['opacity'].max():.4f}]")
print(f"  scaling: range=[{gaussians_loaded['scaling'].min():.6f}, {gaussians_loaded['scaling'].max():.6f}]")
print(f"  color: range=[{gaussians_loaded['color'].min():.4f}, {gaussians_loaded['color'].max():.4f}]")
print(f"  xyz: range=[{gaussians_loaded['xyz'].min():.4f}, {gaussians_loaded['xyz'].max():.4f}]")

# 6. xyz 坐标范围 vs 渲染参数
print("\n--- 坐标范围 vs 渲染参数 ---")
xyz = gaussians_loaded["xyz"]
print(f"  xyz range: x=[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] z=[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")
print(f"  渲染焦距: fx=211, fy=211 (384x256)")
print(f"  预期像素坐标: u = fx * x/z + cx")
z_median = xyz[:, 2].median()
print(f"  z 中位数: {z_median:.2f}")
print(f"  预期 u 范围: {211 * xyz[:,0].min()/z_median + 192:.0f} ~ {211 * xyz[:,0].max()/z_median + 192:.0f}")
print(f"  预期 v 范围: {211 * xyz[:,1].min()/z_median + 128:.0f} ~ {211 * xyz[:,1].max()/z_median + 128:.0f}")

# 7. 问题总结
print("\n" + "=" * 70)
print("问题总结")
print("=" * 70)
print(f"""
1. scaling 格式:
   - benchmark_cpu 输出的 scaling 已经是激活后的值 (exp 后)
   - 范围: [{scaling.min():.6f}, {scaling.max():.6f}], 中位数 {scaling.median():.6f}
   - render_cpu_alpha 直接用这个值作为屏幕空间高斯半径(像素)
   - 但这些值非常小 (~{scaling.median():.4f}),只有几个像素
   - render_cpu_multiview 默认再乘 scale_modifier=0.55,更小
   - 结果: 高斯太小,画面出现稀疏点而非连续表面

2. features_dc 格式:
   - benchmark_cpu 输出的是 SH DC 系数 (未加 0.5)
   - render_cpu_alpha 做了 color = (0.5 + SH_C0 * dc).clamp(0,1)
   - SH_C0 = 0.28209, 所以 color = 0.5 + 0.28 * dc
   - 这个转换是正确的

3. opacity 格式:
   - benchmark_cpu 输出的是 sigmoid 后的值 [0,1]
   - render_cpu_alpha 直接用,正确

4. rotation 格式:
   - benchmark_cpu 输出的是 normalize 后的四元数
   - render_cpu_alpha 直接用,正确

5. 核心问题: scaling 太小
   - Flash3D 的 scaling 经过 exp(scale_scale * raw + log(scale_bias))
   - scale_scale=0.1, scale_bias=0.02
   - 最终 scaling ≈ exp(0.1 * raw) * 0.02, 范围很小
   - render_cpu_alpha 需要的是屏幕空间的像素半径
   - 两者量纲不匹配!
""")
