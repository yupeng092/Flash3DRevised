import torch

data = torch.load("outputs/benchmark_inference/gaussians.pt", map_location="cpu", weights_only=False)
g = data["gaussians"]
n = g["xyz"].shape[0]
meta = data["metadata"]

print("=== Flash3D 推理生成的高斯椭球体数量 ===")
print()
print(f"总高斯数: {n:,}")
print(f"每像素高斯层数: {meta['gaussians_per_pixel']}")
print(f"输入图像尺寸: {meta['input_size_hw'][1]}x{meta['input_size_hw'][0]} (WxH)")
print(f"填充后尺寸: {meta['padded_size_hw'][1]}x{meta['padded_size_hw'][0]} (WxH)")
print()

ph, pw = meta["padded_size_hw"]
gpp = meta["gaussians_per_pixel"]
expected = ph * pw * gpp
print(f"计算公式: 填充后高度 x 填充后宽度 x 每像素层数")
print(f"  = {ph} x {pw} x {gpp}")
print(f"  = {expected:,}")
print(f"实际生成: {n:,}")
print(f"匹配: {'是' if n == expected else '否'}")
print()

# 增强后的数量
data2 = torch.load("outputs/benchmark_enhanced/gaussians.pt", map_location="cpu", weights_only=False)
n2 = data2["gaussians"]["xyz"].shape[0]
print(f"增强后高斯数: {n2:,} (2x 密度)")
print(f"  原始: {n:,} -> 增强: {n2:,}")
