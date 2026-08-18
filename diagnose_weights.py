import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("third_party/unidepth_offline").resolve()))

# 1. 读配置
with open("third_party/unidepth_offline/configs/config_v1_vitl14.json") as f:
    config = json.load(f)

print("=== 配置分析 ===")
print(f"pixel_encoder.name: {config['model']['pixel_encoder']['name']}")
print(f"pixel_encoder.pretrained: {config['model']['pixel_encoder']['pretrained']}")
print(f"num_register_tokens: {config['model']['pixel_encoder'].get('num_register_tokens', '未设置(默认0)')}")
print(f"use_norm: {config['model']['pixel_encoder'].get('use_norm', '未设置(默认False)')}")

# 2. 构建模型看实际参数
print("\n=== 构建模型 ===")
from unidepth.models import UniDepthV1
model = UniDepthV1(config)

# 检查 pixel_encoder 的关键属性
pe = model.pixel_encoder
print(f"pixel_encoder 类型: {type(pe).__name__}")
print(f"num_register_tokens: {pe.num_register_tokens}")
print(f"use_norm: {pe.use_norm}")
print(f"register_tokens shape: {pe.register_tokens.shape}")
print(f"norm 类型: {type(pe.norm).__name__}")
print(f"norm 参数: weight={pe.norm.weight.shape}, bias={pe.norm.bias.shape}")

# 3. 检查 forward 是否真的跳过了 register_tokens 和 norm
print("\n=== forward 行为分析 ===")
print(f"num_register_tokens={pe.num_register_tokens}, 所以 register_tokens 在 forward 中{'会' if pe.num_register_tokens else '不会'}被使用")
print(f"use_norm={pe.use_norm}, 所以 norm 在 forward 中{'会' if pe.use_norm else '不会'}被使用")

# 4. 加载权重看缺失情况
print("\n=== 加载权重 ===")
ckpt = torch.load("weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin", map_location="cpu", weights_only=False)
info = model.load_state_dict(ckpt, strict=False)
print(f"missing_keys: {info.missing_keys}")
print(f"unexpected_keys: {info.unexpected_keys}")

# 5. 检查缺失的 key 是否影响 forward
print("\n=== 缺失 key 影响分析 ===")
for k in info.missing_keys:
    if "register_tokens" in k:
        print(f"  {k}: 不影响 (num_register_tokens=0, forward 跳过)")
    elif "norm.weight" in k or "norm.bias" in k:
        print(f"  {k}: 不影响 (use_norm=False, forward 跳过)")
    else:
        print(f"  {k}: ⚠️ 可能影响!")
