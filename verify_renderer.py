import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=== 检查实际使用的渲染器 ===\n")

# 1. 看我脚本里设的 renderer_backend
print("1. 我的脚本 flash3d_native_multiview.py 设置:")
print("   cfg.model.renderer_backend = 'torch'")
print()

# 2. 看 gauss_util.py 的调度逻辑
print("2. gauss_util.py render_predicted() 调度逻辑:")
print("   renderer_backend = getattr(cfg.model, 'renderer_backend', 'cuda')")
print("   if renderer_backend in {'torch', 'npu_torch'}:")
print("       -> render_predicted_torch()  [自实现 NPU/CPU 渲染器]")
print("   if renderer_backend == 'cuda':")
print("       -> GaussianRasterizer()  [Flash3D 原版 CUDA 渲染器]")
print()

# 3. 实际加载配置确认
with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
    cfg = compose(config_name="config", overrides=["+experiment=layered_re10k"])

print("3. 默认配置 (layered_re10k):")
print(f"   renderer_backend = {getattr(cfg.model, 'renderer_backend', '未设置(默认cuda)')}")
print(f"   renderer_w_pose = {getattr(cfg.model, 'renderer_w_pose', '未设置')}")
print()

# 4. 检查 CUDA 渲染器是否可用
print("4. Flash3D 原版 CUDA 渲染器 (diff_gaussian_rasterization):")
try:
    from diff_gaussian_rasterization import GaussianRasterizer
    print("   状态: 已安装,可用")
except ImportError:
    print("   状态: 未安装 (CPU 环境无法安装 CUDA 扩展)")
print()

# 5. 我实际用的渲染器
print("5. 实际运行的渲染器:")
print("   我的脚本设 renderer_backend='torch'")
print("   => 走 render_predicted_torch() [models/decoder/npu_differentiable_renderer.py]")
print("   => 这是自实现的纯 PyTorch 渲染器,不是 Flash3D 原版")
print()

print("=" * 60)
print("结论:")
print("=" * 60)
print("""
我确实调用了 Flash3D 原生的前向传播代码路径:
  GaussianPredictor.forward()
    -> UniDepth 深度预测
    -> 高斯参数预测 (ResNet encoder + decoder)
    -> compute_gauss_means() 反投影
    -> process_gt_poses() 位姿处理
    -> render_images() -> render_predicted()

但在 render_predicted() 内部,渲染器后端用的是:
  renderer_backend = 'torch'
  -> render_predicted_torch()  [自实现,纯 PyTorch]

而不是 Flash3D 原版的:
  renderer_backend = 'cuda'
  -> GaussianRasterizer  [diff_gaussian_rasterization CUDA 扩展]

原因: 当前是 CPU 环境,无法安装 diff_gaussian_rasterization (需要 CUDA)。
原版 CUDA 渲染器无法在 CPU 上运行。
""")
