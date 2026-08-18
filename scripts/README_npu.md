# 单卡 Ascend 910B 预训练流程

所有命令均在 Flash3D 项目根目录执行，并假定已安装 CANN 和与之匹配的 `torch_npu`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/check_npu_env.py
python scripts/test_npu_renderer.py
```

编码器为 **Depth Anything V1（官方相对深度），预训练时冻结**。源码和权重都从
HuggingFace `LiheYoung/Depth-Anything`（space）获取，无需访问 GitHub。两个冻结变体可选：

| 变体 | 实验配置 | encoder | 权重文件 | 参数量 | 大小 |
| --- | --- | --- | --- | --- | --- |
| ViT-B（默认） | `+experiment=layered_re10k_npu` | `vitb` | `depth_anything_vitb14.pth` | ~97.5M | ~371 MB |
| ViT-S（省显存） | `+experiment=layered_re10k_npu_vits` | `vits` | `depth_anything_vits14.pth` | ~24.8M | ~96 MB |

### 下载源码与权重

**源码**（DA V1 + 内置 DINOv2 主干，约 22 个文件）：

```bash
python scripts/download_depth_anything_v1.py --probe-only   # 先看仓库结构
# 源码通过 huggingface_hub snapshot_download 获取，见下方脚本或手动：
python -c @"
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
from pathlib import Path
dst = Path('third_party/Depth-Anything'); dst.mkdir(parents=True, exist_ok=True)
snapshot_download('LiheYoung/Depth-Anything', repo_type='space', local_dir=str(dst),
    allow_patterns=['depth_anything/**','torchhub/facebookresearch_dinov2_main/**',
                    'requirements.txt','README.md'],
    ignore_patterns=['**/.DS_Store','**/*.md','**/LICENSE','**/conda.yaml',
                     '**/requirements*.txt','**/setup.*','**/scripts/**','**/run/**',
                     '**/eval/**','**/data/**','**/train/**','**/loss/**',
                     '**/logging/**','**/distributed/**','**/fsdp/**','**/configs/**'])
"@
```

> 必须同时获取 `depth_anything/`（DPT 模型定义）和 `torchhub/facebookresearch_dinov2_main/`
> （DINOv2 主干源码，`dpt.py` 通过 `torch.hub.load(source='local')` 离线加载它）。

**权重**（含 DINOv2 主干的完整 state dict）：

```bash
# 下载 ViT-B（默认，~371 MB）
python scripts/download_depth_anything_v1.py

# 下载 ViT-S（~96 MB）
python scripts/download_depth_anything_v1.py --encoders vits

# 同时下载 vits + vitb
python scripts/download_depth_anything_v1.py --encoders vits vitb
```

> 若 `huggingface.co` 直连不通（如 GFW），脚本自动切换 `hf-mirror.com` 镜像；
> 也可手动 `export HF_ENDPOINT=https://hf-mirror.com`。

### 验证安装

```bash
python scripts/test_cpu_smoke.py
```
该脚本在 CPU 上验证三层：Hydra 配置链路、可微渲染器前向+反向、DA V1 编码器构造+权重加载+冻结+前向。

> 冻结意味着：DA 编码器 `requires_grad_(False)`、不进优化器 param groups、推理走
> `torch.no_grad()`。只有 ResNet 主干 + Gaussian 解码器被训练。
>
> V1 为相对深度，输出按图 min-max 归一化到 `[0, 1]`。Flash3D 在训练时通过
> `train.scale_pose_by_depth` + `misc.depth.estimate_depth_scale` 从稀疏 COLMAP 点云
> 恢复绝对尺度，因此相对深度先验足以支撑单图前馈训练。若要回退到 V2 metric 先验，
> 使用 `+experiment=layered_re10k_depth_anything_v2`（或显式
> `model.name=depth_anything_v2` 且 `model/depth=depth_anything_v2`）。

## 可选：预计算冻结深度

这一步减少训练时的 NPU 延迟和显存。输出目录可随后作为 `dataset.depth_path` 使用。
`--version` 默认为 `v1`；预计算 V2 metric 深度时传 `--version v2` 并按需覆盖
`--source-dir` / `--checkpoint`。

ViT-B（默认）：
```bash
python scripts/precompute_depth_anything_re10k.py \
  --version v1 --encoder vitb \
  --checkpoint weights/depth-anything/depth_anything_vitb14.pth \
  --data-path /datasets/RealEstate10K \
  --output /datasets/RealEstate10K-depth-anything-vitb \
  --device npu
```

ViT-S（省显存）：
```bash
python scripts/precompute_depth_anything_re10k.py \
  --version v1 --encoder vits \
  --checkpoint weights/depth-anything/depth_anything_vits14.pth \
  --data-path /datasets/RealEstate10K \
  --output /datasets/RealEstate10K-depth-anything-vits \
  --device npu
```

## 实测峰值显存

```bash
python scripts/profile_npu_pretrain.py \
  --data-path /datasets/RealEstate10K \
  --depth-path /datasets/RealEstate10K-depth-anything-vitb
```

若峰值超过目标，先将 `model.npu_renderer_max_gaussians` 从 65536 下调至 32768，
再降低每 tile 的上限；不要先增大 batch size。

## 单卡训练与评估

ViT-B（默认，`EXPERIMENT=layered_re10k_npu`）：
```bash
NPU_ID=0 bash scripts/train_npu.sh \
  dataset.data_path=/datasets/RealEstate10K \
  dataset.preload_depths=true \
  dataset.depth_path=/datasets/RealEstate10K-depth-anything-vitb
```

ViT-S（省显存，`EXPERIMENT=layered_re10k_npu_vits`）：
```bash
EXPERIMENT=layered_re10k_npu_vits NPU_ID=0 bash scripts/train_npu.sh \
  dataset.data_path=/datasets/RealEstate10K \
  dataset.preload_depths=true \
  dataset.depth_path=/datasets/RealEstate10K-depth-anything-vits
```

评估（两个变体共用，按训练时用的实验配置即可）：
```bash
NPU_ID=0 bash scripts/evaluate_npu.sh \
  dataset.data_path=/datasets/RealEstate10K
```

两个实验配置均使用 Depth Anything V1（相对深度 + scale 恢复）、**冻结编码器**、
单卡和 `npu_torch` 可微高斯渲染器。它们不是多卡 HCCL 启动配置。
