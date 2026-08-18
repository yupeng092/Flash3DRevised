from PIL import Image
import json

print("=== Resolution comparison ===")
im = Image.open("outputs/courtyard_colmap_final/rgb/view_000.png")
print(f"Rendered: {im.size[0]}x{im.size[1]}")

im2 = Image.open(r"D:\Python Project\courtyard\images\dslr_images_undistorted\DSC_0286.JPG")
print(f"Original: {im2.size[0]}x{im2.size[1]}")
print(f"Scale factor: {im2.size[0]/im.size[0]:.1f}x downscaled")

with open("outputs/courtyard_colmap_s2/render_poses.json") as f:
    poses = json.load(f)
p = poses[0]
print(f"\nCOLMAP camera: {p['width']}x{p['height']}, fx={p['fx']:.1f}")
print(f"Rendered at 384x256 -> fx scaled to {p['fx']*384/p['width']:.1f}")
