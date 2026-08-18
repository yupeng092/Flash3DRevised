from PIL import Image
import os

print("=== High-res render results ===")
for f in sorted(os.listdir("outputs/courtyard_colmap_hires/rgb")):
    im = Image.open(f"outputs/courtyard_colmap_hires/rgb/{f}")
    kb = os.path.getsize(f"outputs/courtyard_colmap_hires/rgb/{f}") // 1024
    print(f"{f}: {im.size[0]}x{im.size[1]}, {kb}KB")

print()
print("=== Original ===")
im = Image.open(r"D:\Python Project\courtyard\images\dslr_images_undistorted\DSC_0286.JPG")
print(f"Original: {im.size[0]}x{im.size[1]}")
print(f"Render scale: {im.size[0]/3072:.1f}x (half resolution)")
