import torch
import numpy as np
import json
from pathlib import Path

print("=== 坐标系和尺度分析 ===\n")

# 1. Flash3D 点云 (camera 0 坐标系)
d = torch.load("outputs/courtyard_benchmark/gaussians.pt", map_location="cpu", weights_only=True)
g = d["gaussians"]
xyz_flash3d = g["xyz"].numpy()
print(f"Flash3D 点云 (camera 0 坐标系):")
print(f"  点数: {len(xyz_flash3d)}")
print(f"  x: [{xyz_flash3d[:,0].min():.2f}, {xyz_flash3d[:,0].max():.2f}]")
print(f"  y: [{xyz_flash3d[:,1].min():.2f}, {xyz_flash3d[:,1].max():.2f}]")
print(f"  z: [{xyz_flash3d[:,2].min():.2f}, {xyz_flash3d[:,2].max():.2f}]")
print(f"  深度中位数: {np.median(xyz_flash3d[:,2]):.2f}")

# 2. COLMAP 点云 (世界坐标系)
colmap_pts = []
with open(r"D:\Python Project\courtyard\dslr_calibration_undistorted\points3D.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        colmap_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
colmap_pts = np.array(colmap_pts)
print(f"\nCOLMAP 点云 (世界坐标系):")
print(f"  点数: {len(colmap_pts)}")
print(f"  x: [{colmap_pts[:,0].min():.2f}, {colmap_pts[:,0].max():.2f}]")
print(f"  y: [{colmap_pts[:,1].min():.2f}, {colmap_pts[:,1].max():.2f}]")
print(f"  z: [{colmap_pts[:,2].min():.2f}, {colmap_pts[:,2].max():.2f}]")

# 3. COLMAP 相机位姿
images = []
with open(r"D:\Python Project\courtyard\dslr_calibration_undistorted\images.txt") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        name = parts[9]
        images.append({"id": img_id, "qvec": [qw,qx,qy,qz], "tvec": [tx,ty,tz], "name": name})
        i += 1  # skip points2d line

# 找 DSC_0286 (camera 0)
cam0 = None
for img in images:
    if "DSC_0286" in img["name"]:
        cam0 = img
        break
print(f"\nCamera 0 (DSC_0286) COLMAP 位姿:")
print(f"  qvec: {cam0['qvec']}")
print(f"  tvec: {cam0['tvec']}")

# 4. 计算 camera 0 到其他相机的相对位姿
def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])

R0 = quat_to_rot(cam0["qvec"])
t0 = np.array(cam0["tvec"])

# COLMAP 位姿是 world-to-camera: P_cam = R @ P_world + t
# Camera 0: P_cam0 = R0 @ P_world + t0
# 所以: P_world = R0^T @ (P_cam0 - t0)
# Camera i: P_cami = Ri @ P_world + ti = Ri @ R0^T @ (P_cam0 - t0) + ti
#         = Ri @ R0^T @ P_cam0 - Ri @ R0^T @ t0 + ti
# 相对变换: P_cami = R_rel @ P_cam0 + t_rel
#   R_rel = Ri @ R0^T
#   t_rel = ti - Ri @ R0^T @ t0 = ti - R_rel @ t0

print(f"\n=== 相对位姿 (camera 0 坐标系) ===")
relative_poses = []
for img in images[:10]:  # 前 10 个
    Ri = quat_to_rot(img["qvec"])
    ti = np.array(img["tvec"])
    R_rel = Ri @ R0.T
    t_rel = ti - R_rel @ t0
    
    # 计算平移距离和旋转角度
    trans_dist = np.linalg.norm(t_rel)
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    
    relative_poses.append({"name": img["name"], "R": R_rel, "t": t_rel, "trans": trans_dist, "angle": angle})
    print(f"  {img['name']}: trans={trans_dist:.2f}, angle={angle:.1f}°")

# 5. 尺度估计
# COLMAP 点云在 camera 0 坐标系下的深度
pts_cam0 = (R0 @ colmap_pts.T).T + t0
colmap_depths = pts_cam0[:, 2]
colmap_depth_median = np.median(colmap_depths[colmap_depths > 0])
flash3d_depth_median = np.median(xyz_flash3d[:, 2])
scale = colmap_depth_median / flash3d_depth_median

print(f"\n=== 尺度对齐 ===")
print(f"  Flash3D 深度中位数: {flash3d_depth_median:.2f}")
print(f"  COLMAP 深度中位数: {colmap_depth_median:.2f}")
print(f"  尺度因子: {scale:.4f}")
print(f"  Flash3D 点云缩放后深度范围: [{flash3d_depth_median*scale*0.3:.2f}, {flash3d_depth_median*scale*3:.2f}]")
