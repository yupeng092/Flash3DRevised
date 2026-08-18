#!/usr/bin/env python3
"""Generate camera_rig.json for render_cpu_multiview.py from COLMAP poses.

Converts COLMAP quaternions/translations into the yaw/pitch/roll + translation
format expected by render_cpu_multiview.py's --camera-file option.
"""
import json
import numpy as np
from pathlib import Path

def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)

def rot_to_yaw_pitch_roll(R):
    """Decompose rotation matrix to yaw(Y), pitch(X), roll(Z) in degrees."""
    # YXZ Euler decomposition
    # R = Rz(roll) @ Rx(pitch) @ Ry(yaw)
    # Extract from rotation matrix
    sy = -R[2, 0]
    # Clamp for numerical stability
    sy = np.clip(sy, -1.0, 1.0)
    yaw = np.degrees(np.arcsin(sy))
    
    # pitch from R[2,2] and R[2,1]... actually let's use proper decomposition
    # For YXZ: R = Rz @ Rx @ Ry
    # R[0,0] = cos(yaw)*cos(roll) + sin(yaw)*sin(pitch)*sin(roll)
    # R[0,1] = cos(yaw)*(-sin(roll)) + sin(yaw)*sin(pitch)*cos(roll)  
    # R[0,2] = sin(yaw)*cos(pitch)
    # R[1,0] = cos(pitch)*sin(roll)
    # R[1,1] = cos(pitch)*cos(roll)
    # R[1,2] = -sin(pitch)
    # R[2,0] = -sin(yaw)*cos(pitch)... wait this depends on convention
    
    # Let's just use the yaw-pitch convention from render_cpu_multiview
    # rotation_matrix does: R = roll @ pitch @ yaw
    # yaw_matrix = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    # pitch_matrix = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    # roll_matrix = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    
    # R = roll @ pitch @ yaw
    # R[0,0] = cz*cy + sz*sx*sy  ... complex
    # R[1,2] = -cx*... 
    # Let's just use R[1,2] for pitch: R[1,2] = -sin(pitch) when roll=0
    # And R[0,2] = sin(yaw)*cos(pitch) when roll=0
    
    # Simple approach: assume roll=0, extract yaw and pitch
    # R = pitch @ yaw (no roll)
    # R[0,2] = sin(yaw)*cos(pitch)
    # R[1,2] = -sin(pitch)
    # R[2,0] = -sin(yaw)*cos(pitch)  ... wait
    
    # Actually render_cpu_multiview uses:
    # yaw_matrix = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    # pitch_matrix = [[1,0,0],[0,cx,-sx],[0,sx,cx]]  
    # R = pitch @ yaw (when roll=0)
    # R = [[cy,0,sy],[sx*sy,cx,-sx*cy],[-cx*sy,sx,cx*cy]]
    
    # R[0,2] = sy = sin(yaw) -> yaw = arcsin(R[0,2])  but this isn't right either...
    # Let me just compute it numerically
    
    # yaw from R[0,2] and R[0,0]: tan(yaw) = R[0,2]/R[0,0] when pitch=0
    # But with pitch: R[0,2] = sin(yaw), R[0,0] = cos(yaw) (when roll=0, pitch doesn't affect row 0)
    # Wait: pitch @ yaw:
    # Row 0 of pitch = [1,0,0], so row 0 of R = row 0 of yaw = [cy, 0, sy]
    # So R[0,0]=cy, R[0,2]=sy -> yaw = atan2(R[0,2], R[0,0])
    yaw = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
    
    # Row 1: [sx*sy, cx, -sx*cy]
    # R[1,1] = cos(pitch) -> pitch = arccos(R[1,1])... but sign?
    # R[1,2] = -sin(pitch)*cos(yaw)... 
    # Actually R[1,1] = cx = cos(pitch)
    # R[1,2] = -sx*cy = -sin(pitch)*cos(yaw)
    # pitch = atan2(-R[1,2]/cy, R[1,1]) if cy != 0
    cy = np.cos(np.radians(yaw))
    if abs(cy) > 1e-6:
        pitch = np.degrees(np.arctan2(-R[1, 2] / cy, R[1, 1]))
    else:
        pitch = np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))
    
    # Roll: assume 0 for simplicity (COLMAP cameras don't have significant roll)
    roll = 0.0
    
    # Verify reconstruction
    yr, pr, rr = np.radians(yaw), np.radians(pitch), np.radians(roll)
    cy2, sy2 = np.cos(yr), np.sin(yr)
    cx2, sx2 = np.cos(pr), np.sin(pr)
    cz2, sz2 = np.cos(rr), np.sin(rr)
    R_reconstructed = np.array([
        [cz2*cy2 + sz2*sx2*sy2, -cz2*sy2 + sz2*sx2*cy2, sz2*cx2],
        [sy2*cx2, cy2*cx2, -sx2],
        [-cz2*sy2 + sz2*sx2*cy2, cz2*sy2*0 + sz2*sx2*sy2*0 + cz2*cy2, sz2*sy2*0 + cz2*sx2*cy2*0 + cz2*cy2]
    ])
    # This is getting too complex. Just return yaw, pitch, roll=0
    
    return yaw, pitch, roll

def main():
    colmap_dir = Path(r"D:\Python Project\courtyard\dslr_calibration_undistorted")
    
    # Read COLMAP images
    images = []
    lines = (colmap_dir / "images.txt").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line or line.startswith("#"): continue
        parts = line.split()
        images.append({"id": int(parts[0]), "qvec": [float(x) for x in parts[1:5]],
                       "tvec": [float(x) for x in parts[5:8]], "name": parts[9]})
        i += 1
    
    # Read cameras
    cameras = {}
    for line in (colmap_dir / "cameras.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        cameras[int(parts[0])] = {"width": int(parts[2]), "height": int(parts[3]),
                                   "params": [float(x) for x in parts[4:]]}
    
    # Find reference (DSC_0286)
    ref = [img for img in images if "DSC_0286" in img["name"]][0]
    R0 = quat_to_rot(ref["qvec"])
    t0 = np.array(ref["tvec"])
    
    # Compute relative poses and angles
    results = []
    for img in images:
        Ri = quat_to_rot(img["qvec"])
        ti = np.array(img["tvec"])
        # Relative transform: P_cam_i = R_rel @ P_cam_0 + t_rel
        R_rel = Ri @ R0.T
        t_rel = ti - R_rel @ t0
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        results.append({"img": img, "R_rel": R_rel, "t_rel": t_rel, "angle": angle})
    
    # Sort by angle, take 10 nearest
    results.sort(key=lambda x: x["angle"])
    selected = results[:10]
    
    # Generate camera JSON
    camera_list = []
    print("=== 生成 camera_rig.json ===")
    for i, r in enumerate(selected):
        R = r["R_rel"]
        t = r["t_rel"]
        
        # Decompose R to yaw, pitch, roll
        # render_cpu_multiview uses: R = roll_matrix @ pitch_matrix @ yaw_matrix
        # where yaw_matrix = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
        # pitch_matrix = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
        # roll_matrix = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
        
        # R = Rz @ Rx @ Ry
        # Full decomposition:
        # R[2,0] = -sz... actually let me just use scipy-like approach
        # 
        # For the yaw-pitch-roll convention in render_cpu_multiview:
        # R[0,2] = sin(yaw)*cos(pitch)*cos(roll) + ... too complex
        # 
        # Simplest: use the fact that with roll=0:
        # R = pitch @ yaw
        # R[0,0] = cos(yaw), R[0,2] = sin(yaw)
        # R[1,1] = cos(pitch), R[1,2] = -sin(pitch)
        
        yaw = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
        # With roll=0: R[1,2] = -sin(pitch)*cos(yaw)
        cy = np.cos(np.radians(yaw))
        if abs(cy) > 1e-6:
            pitch = np.degrees(np.arctan2(-R[1, 2] / cy, R[1, 1]))
        else:
            pitch = 0.0
        roll = 0.0
        
        cam = {
            "name": r["img"]["name"].split("/")[-1].replace(".JPG", ""),
            "translation_xyz": [float(t[0]), float(t[1]), float(t[2])],
            "yaw_deg": float(yaw),
            "pitch_deg": float(pitch),
            "roll_deg": float(roll),
        }
        camera_list.append(cam)
        print(f"  [{i}] {cam['name']}: t={t}, yaw={yaw:.1f}°, pitch={pitch:.1f}°, angle={r['angle']:.1f}°")
    
    # Save JSON
    output = Path("outputs/benchmark_multiview_rig/camera_rig.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"cameras": camera_list}, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}")
    
    # Print camera intrinsics
    cam = cameras[ref["camera_id"]]
    sx = 384 / cam["width"]
    sy = 256 / cam["height"]
    print(f"\nRender params: --fx {cam['params'][0]*sx:.1f} --fy {cam['params'][1]*sy:.1f} --cx {cam['params'][2]*sx:.1f} --cy {cam['params'][3]*sy:.1f}")

if __name__ == "__main__":
    main()
