import json
import numpy as np

with open("outputs/courtyard_flash3d_multiview/render_poses.json") as f:
    poses = json.load(f)

print("=== 10 个视角的覆盖情况 ===")
print(f"{'视角':<12} {'平移':>8} {'旋转角':>8} {'可见高斯':>10} {'alpha':>8} {'能用?':>6}")
print("-" * 60)

results = [
    (0, 98968, 0.6743),
    (1, 65935, 0.4869),
    (2, 23047, 0.2048),
    (3, 0, 0.0),
    (4, 0, 0.0),
    (5, 0, 0.0),
    (6, 87722, 0.4265),
    (7, 0, 0.0),
    (8, 98968, 0.1773),
    (9, 96749, 0.0606),
]

for i, (idx, vis, alpha) in enumerate(results):
    p = poses[i]
    usable = "YES" if (vis > 10000 and alpha > 0.15) else "NO"
    print(f"view_{idx:03d}({p['name'][-12:]:>12}) {p['trans']:>8.2f} {p['angle']:>8.1f} {vis:>10,} {alpha:>8.2f} {usable:>6}")

print()
print("=== 原因分析 ===")
print("Flash3D 单图点云只有正面表面(单面重建)")
print("旋转角度 > 30° 的视角看到的是点云背面/侧面 → 没有高斯 → 空白")
print()
print("=== 可用视角(angle < 30°) ===")
usable_indices = [i for i, (_, vis, alpha) in enumerate(results) if vis > 10000 and alpha > 0.15]
print(f"可用: {len(usable_indices)} / 10")
print(f"索引: {usable_indices}")
