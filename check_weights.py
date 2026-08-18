import torch

print("=== UniDepth V1 权重 key 分析 ===\n")

ckpt = torch.load(
    "weights/unidepth-v1-cnvnxtl/unidepth_v1_vitl14.bin",
    map_location="cpu",
    weights_only=False,
)

# 看顶层结构
if isinstance(ckpt, dict):
    print(f"顶层 keys: {list(ckpt.keys())[:20]}")
    if "model" in ckpt:
        state_dict = ckpt["model"]
        print(f"  'model' 下有 {len(state_dict)} 个 key")
    else:
        state_dict = ckpt
        print(f"  直接 state_dict, {len(state_dict)} 个 key")
else:
    state_dict = ckpt
    print(f"类型: {type(ckpt)}")

# 搜索 register_tokens 和 norm 相关的 key
print(f"\n--- register_tokens 相关 ---")
register_keys = [k for k in state_dict if "register_tokens" in k]
for k in register_keys:
    print(f"  {k}: {state_dict[k].shape}")

print(f"\n--- pixel_encoder.norm 相关 ---")
norm_keys = [k for k in state_dict if "pixel_encoder.norm" in k]
for k in norm_keys:
    print(f"  {k}: {state_dict[k].shape}")

print(f"\n--- pixel_encoder 相关 key 总数 ---")
pe_keys = [k for k in state_dict if "pixel_encoder" in k]
print(f"  {len(pe_keys)} keys")
print(f"  前10个:")
for k in pe_keys[:10]:
    print(f"    {k}: {state_dict[k].shape}")

print(f"\n--- pixel_encoder 最后几个 key ---")
for k in pe_keys[-10:]:
    print(f"    {k}: {state_dict[k].shape}")

# 看 cls_token 和 pos_embed
print(f"\n--- 其他关键 key ---")
for pattern in ["cls_token", "pos_embed", "mask_token"]:
    matches = [k for k in state_dict if pattern in k and "pixel_encoder" in k]
    for k in matches:
        print(f"  {k}: {state_dict[k].shape}")
