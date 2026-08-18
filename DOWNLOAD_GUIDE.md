# 数据集与权重文件下载指南

本仓库（`Flash3DRevised`）已通过 `.gitignore` 排除了所有大文件与大数据目录。
在服务器上 `git clone` 后，**代码可直接运行**，但训练/推理所需的数据集、
预训练权重、第三方源码需要按本文档单独下载补齐。

> 目录约定：下文所有相对路径均以仓库根目录（`Flash3DRevised/`）为基准。

---

## 0. 目录速查表

| 本地路径 | 用途 | 大小(参考) | 是否必需 | 来源 |
|---|---|---|---|---|
| `exp/re10k_v2/checkpoints/model_re10k_v2.pth` | Flash3D 预训练模型 | ~658 MB | 评估/推理必需 | HuggingFace `einsafutdinov/flash3d` |
| `weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin` | UniDepth V1 ViT-L/14 深度模型 | ~1.32 GB | UniDepth 配置必需 | HuggingFace `Miyanishi/UniDepth` |
| `weights/depth-anything/depth_anything_vit{b,s,l}14.pth` | Depth Anything V1 权重 | vits~96 / vitb~372 / vitl~1280 MB | DA-V1 配置必需 | HuggingFace `LiheYoung/Depth-Anything` |
| `weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth` | Depth Anything V2 权重 | ~465 MB | DA-V2 配置必需 | GitHub `DepthAnything/Depth-Anything-V2` |
| `weights/realesrgan/RealESRGAN_x4.pth` | Real-ESRGAN 超分模型 | ~64 MB | 超分脚本必需 | GitHub `xinntao/Real-ESRGAN` |
| `weights/torch_hub_cache/hub/checkpoints/vgg16-397923af.pth` | VGG16(torchvision自动下载) | ~528 MB | 感知损失可选 | torch hub |
| `data/RealEstate10K/` | RE10K 训练/测试数据 | 数十~数百 GB | 训练必需 | YouTube + 官方元数据 |
| `data/re10k_hf_raw/` | RE10K HF 测试集(.torch) | ~1 GB | CPU 调试可选 | HuggingFace `Hualingchu/RealEstate10K_test` |
| `third_party/Depth-Anything/` | DA-V1 模型源码 | ~几 MB | DA-V1 配置必需 | GitHub `LiheYoung/Depth-Anything` |
| `third_party/Depth-Anything-V2/` | DA-V2 模型源码 | ~几 MB | DA-V2 配置必需 | GitHub `DepthAnything/Depth-Anything-V2` |
| `third_party/unidepth_offline/` | UniDepth 模型源码 | ~几 MB | UniDepth 配置必需 | GitHub `Miyanishi/UniDepth` |
| `pretrain_dataset/` | 自采集预训练视频(mp4) | ~4.3 GB(本机) | 自定义预训练可选 | Pexels/Mixkit 爬取脚本 |

---

## 1. Flash3D 预训练模型（评估/推理必需）

仓库自带脚本 `misc/download_pretrained_models.py`，从 HuggingFace 下载：

```bash
# 下载到 exp/re10k_v2/checkpoints/model_re10k_v2.pth (~658 MB)
python -m misc.download_pretrained_models -o exp/re10k_v2
```

- **HuggingFace 仓库**：`einsafutdinov/flash3d`
- **文件**：`model_re10k_v2.pth`
- **目标路径**：`exp/re10k_v2/checkpoints/model_re10k_v2.pth`

下载后即可评估：

```bash
sh evaluate.sh exp/re10k_v2
```

> 国内服务器若 huggingface.co 不可达，设置镜像：
> `export HF_ENDPOINT=https://hf-mirror.com`

---

## 2. 深度模型权重

Flash3D 支持三种深度先验，按你使用的配置（`model.depth.name`）下载对应权重。

### 2a. UniDepth V1（默认配置 `layered_re10k` / `layered_kitti` / `layered_nyuv2`）

- **目标路径**：`weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin` (~1.32 GB)
- **来源**：HuggingFace `Miyanishi/UniDepth`，文件 `checkpoints/unidepth_v1_cnvnxtl/unidepth_v1_vitl14.bin`

```bash
# 方式一：huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('Miyanishi/UniDepth', \
  'checkpoints/unidepth_v1_cnvnxtl/unidepth_v1_vitl14.bin', \
  local_dir='weights/unidepth-v1-cnvnxtl')"

# 方式二：直接 wget（注意路径映射）
mkdir -p weights/unidepth-v1-cnvnxtl
wget -O weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin \
  https://huggingface.co/Miyanishi/UniDepth/resolve/main/checkpoints/unidepth_v1_cnvnxtl/unidepth_v1_vitl14.bin
```

**同时需要 UniDepth 源码**（见第 4 节 `third_party/unidepth_offline`）。

### 2b. Depth Anything V1（配置 `layered_re10k_depth_anything_v1` / NPU 预训练）

仓库自带脚本 `scripts/download_depth_anything_v1.py`，自动探测 HF 端点并下载：

```bash
# 默认下载 vitb (~372 MB)
python scripts/download_depth_anything_v1.py

# 下载多个 encoder
python scripts/download_depth_anything_v1.py --encoders vits vitb

# 仅探测仓库内容不下载
python scripts/download_depth_anything_v1.py --probe-only
```

- **HuggingFace 仓库**：`LiheYoung/Depth-Anything`（space 类型）
- **目标路径**：`weights/depth-anything/depth_anything_vit{b,s,l}14.pth`
  - `vits` → ~96 MB
  - `vitb` → ~372 MB
  - `vitl` → ~1280 MB
- 配置见 `configs/model/depth/depth_anything_v1.yaml`，`encoder` 字段决定使用哪个。

> 脚本已内置 hf-mirror.com 回退，国内可直接用。手动指定：
> `export HF_ENDPOINT=https://hf-mirror.com`

**同时需要 DA-V1 源码**（见第 4 节 `third_party/Depth-Anything`）。

### 2c. Depth Anything V2（配置 `layered_re10k_cpu_debug` / `layered_re10k_depth_anything_v2`）

- **目标路径**：`weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth` (~465 MB)
- **来源**：GitHub `DepthAnything/Depth-Anything-V2` 仓库的 `metric_depth` 权重
- **官方下载页**：https://github.com/DepthAnything/Depth-Anything-V2#24-metric-depth-model-zoo

```bash
mkdir -p weights/depth-anything-v2
# 从官方 release 下载 metric vkitti vitb 权重
wget -O weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hybrid-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth
```

配置见 `configs/model/depth/depth_anything_v2.yaml`。

**同时需要 DA-V2 源码**（见第 4 节 `third_party/Depth-Anything-V2`）。

---

## 3. RealEstate10K 数据集（训练必需）

### 3a. 完整 RE10K（官方流程，训练用）

遵循 [Behind The Scenes](https://github.com/Brummi/BehindTheScenes#-datasets) 流程：

**第 1 步：下载视频元数据（相机位姿）**

从 https://google.github.io/realestate10k/download.html 下载 `test.txt` 和 `train.txt`，
解压到 `data/RealEstate10K/`，目录结构：

```
data/RealEstate10K/train/*.txt
data/RealEstate10K/test/*.txt
```

**第 2 步：下载 YouTube 视频并抽帧**

```bash
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
```

- 依赖 `pytubefix`、`ffmpeg`。
- ⚠️ 此步骤需访问 YouTube，**耗时数天**，且部分视频可能失效。
- 最终生成 `data/RealEstate10K/{train,test}/{seq}/{ts}.jpg`。

**第 3 步：下载 COLMAP 稀疏点云缓存（深度缩放因子估计必需）**

```bash
sh datasets/download_realestate10k_colmap.sh
```

该脚本从 `https://thor.robots.ox.ac.uk/flash3d/` 下载以下文件到 `data/RealEstate10K/`：

| 文件 | 说明 |
|---|---|
| `test.pickle.gz` | 测试集元数据 |
| `train.pickle.gz` | 训练集元数据 |
| `pcl.test.tar` | 测试集 COLMAP 稀疏点云 |
| `pcl.train.tar` | 训练集 COLMAP 稀疏点云 |
| `valid_seq_ids.train.pickle.gz` | 有效序列过滤 |
| `SHA512SUMS` | 校验文件 |

下载后自动 `sha512sum -c SHA512SUMS` 校验。

**第 4 步：预处理过滤**

```bash
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

### 3b. RE10K HF 测试集（CPU 调试/快速验证可选）

若只需少量数据做 CPU 调试，可用 HF 上的 `.torch` 测试集：

```bash
# 下载 10 个 .torch 文件 (~1 GB, ~170 序列) 并转换为 Flash3D 格式
python scripts/convert_hf_re10k.py
```

- **HuggingFace dataset**：`Hualingchu/RealEstate10K_test`
- **文件**：`test/{000000..000009}.torch`（每个 ~100 MB）
- **原始缓存**：`data/re10k_hf_raw/test/`
- **转换输出**：`data/RealEstate10K/`（含 pickle.gz / jpg / pcl.tar）
- 脚本已内置 `HF_ENDPOINT=https://hf-mirror.com`，国内可直接用。

> ⚠️ 此方式生成的 COLMAP 点云是**合成的**（用真实位姿散布的点），
> 仅适合 CPU 训练验证，不能替代完整 RE10K COLMAP 缓存。

---

## 4. 第三方模型源码（third_party/）

部分深度模型的 **模型定义代码** 不在 pip 包里，需要 clone 到 `third_party/`。
这些目录已被 `.gitignore` 排除，不会随仓库分发。

```bash
mkdir -p third_party
cd third_party

# Depth Anything V1 源码（DA-V1 配置必需）
git clone https://github.com/LiheYoung/Depth-Anything.git Depth-Anything

# Depth Anything V2 源码（DA-V2 配置必需）
git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2

# UniDepth 源码（UniDepth 配置必需）
git clone https://github.com/Miyanishi/UniDepth.git unidepth_offline
# UniDepth 代码里通过 torch.hub.load 本地加载，目录名必须为 unidepth_offline

cd ..
```

### 验证

代码会在加载时检查源码是否存在，缺失会报明确错误，例如：

```
FileNotFoundError: Depth Anything V1 source is missing at third_party/Depth-Anything.
  Clone https://github.com/LiheYoung/Depth-Anything there.
```

---

## 5. Real-ESRGAN 超分权重（超分脚本可选）

用于 `realesrgan_upscale.py` / `sr_postprocess.py` 等超分后处理脚本。

- **目标路径**：`weights/realesrgan/RealESRGAN_x4.pth` (~64 MB)
- **来源**：GitHub `xinntao/Real-ESRGAN` releases

```bash
mkdir -p weights/realesrgan
wget -O weights/realesrgan/RealESRGAN_x4.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

> 代码 `sr_postprocess.py` 会依次查找以下文件名（任一即可）：
> `RealESRGAN_x4.pth` / `realesrgan_x4plus.pth` /
> `realesrgan_x4plus.safetensors` / `RealESRGAN_x4.safetensors` /
> `RealESRGAN_x2.pth` / `realesrgan_x2plus.pth`

---

## 6. VGG16（torch hub，感知损失可选）

`weights/torch_hub_cache/hub/checkpoints/vgg16-397923af.pth` (~528 MB)

由 torchvision 通过 `torch.hub.load_state_dict_from_url` 自动下载。
若服务器无网络，可手动放置：

```bash
mkdir -p weights/torch_hub_cache/hub/checkpoints
wget -O weights/torch_hub_cache/hub/checkpoints/vgg16-397923af.pth \
  https://download.pytorch.org/models/vgg16-397923af.pth
# 并设置环境变量指向缓存目录
export TORCH_HOME=weights/torch_hub_cache
```

---

## 7. 预训练视频数据集（自定义预训练，可选）

`pretrain_dataset/` 存放自采集的航拍/户外视频（mp4），用于自定义预训练。
本机约有 307 个 mp4，总计 ~4.3 GB。**这些不是官方数据，按需自行采集。**

仓库提供多个爬取脚本（依赖 `curl_cffi`、`requests`）：

```bash
# Pexels 航拍视角视频（需绕过 Cloudflare，用 curl_cffi）
python download_pexels.py          # 搜索 + 下载，默认 40 个
python download_pexels_full.py     # 更大规模采集
python download_mixkit.py          # Mixkit 来源
python download_selected.py        # 选定视频
python scan_more.py                # 扩展搜索
```

- 输出目录：`pretrain_dataset/`
- 搜索关键词见各脚本顶部 `SEARCH_QUERIES`。
- ⚠️ Pexels/Mixkit 有使用条款限制，请确认合规后再用。

---

## 8. 一键补齐脚本（服务器快速起步）

将以下内容保存为 `setup_data.sh` 并执行（按需注释掉不需要的部分）：

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"

# 国内 HF 镜像（海外服务器可删除此行）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=== [1/6] Flash3D 预训练模型 ==="
python -m misc.download_pretrained_models -o exp/re10k_v2

echo "=== [2/6] Depth Anything V1 (vitb) ==="
python scripts/download_depth_anything_v1.py

echo "=== [3/6] 第三方源码 ==="
mkdir -p third_party && cd third_party
[ -d Depth-Anything ]    || git clone https://github.com/LiheYoung/Depth-Anything.git Depth-Anything
[ -d Depth-Anything-V2 ] || git clone https://github.com/DepthAnything/Depth-Anything-V2.git Depth-Anything-V2
[ -d unidepth_offline ]  || git clone https://github.com/Miyanishi/UniDepth.git unidepth_offline
cd ..

echo "=== [4/6] UniDepth V1 权重 ==="
mkdir -p weights/unidepth-v1-cnvnxtl
[ -f weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin ] || \
  wget -O weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin \
  https://huggingface.co/Miyanishi/UniDepth/resolve/main/checkpoints/unidepth_v1_cnvnxtl/unidepth_v1_vitl14.bin

echo "=== [5/6] Depth Anything V2 权重 ==="
mkdir -p weights/depth-anything-v2
[ -f weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth ] || \
  wget -O weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hybrid-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth

echo "=== [6/6] Real-ESRGAN 权重 ==="
mkdir -p weights/realesrgan
[ -f weights/realesrgan/RealESRGAN_x4.pth ] || \
  wget -O weights/realesrgan/RealESRGAN_x4.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

echo "=== RE10K 数据集（完整，耗时数天） ==="
echo "参见 DOWNLOAD_GUIDE.md 第 3a 节手动执行"
echo "=== 完成 ==="
```

---

## 9. 环境依赖

```bash
# Python 环境
conda create -y python=3.10 -n flash3d
conda activate flash3d

# PyTorch (CUDA 11.8)
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

关键依赖：Python 3.10、PyTorch 2.2.2、CUDA 11.8、xformers 0.0.25.post1、
`huggingface-hub>=0.22.0`、`pytubefix`（RE10K 下载）、`curl_cffi`（Pexels 爬取）。

---

## 10. 常见问题

**Q: huggingface.co 连不上？**
A: 设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`（`download_depth_anything_v1.py`
   和 `convert_hf_re10k.py` 已内置回退，其余脚本需手动设置）。

**Q: YouTube 视频下载失败？**
A: RE10K 部分 YouTube 视频可能已下架。`download_realestate10k.py` 会记录失败序列到
   `failed_videos_{mode}.txt`，后续 `preprocess_realestate10k` 会过滤掉它们。

**Q: UniDepth 加载报 "Missing offline UniDepth source"？**
A: 需 clone UniDepth 源码到 `third_party/unidepth_offline`（见第 4 节）。
   代码 `models/encoder/unidepth_encoder.py` 会检查该路径。

**Q: Depth Anything V1 报 "source is missing at third_party/Depth-Anything"？**
A: 需 clone DA-V1 源码到 `third_party/Depth-Anything`（见第 4 节）。

**Q: 只想做推理评估，需要下载哪些？**
A: 最小集合：① Flash3D 预训练模型（第 1 节）；② UniDepth 权重+源码（第 2a+4 节）；
   ③ RE10K 测试数据（第 3 节，或用 HF 测试集第 3b 节做快速验证）。
