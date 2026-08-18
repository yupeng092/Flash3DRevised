"""
视频背景亮度分析脚本
提取每个视频的多帧，计算平均亮度和暗像素占比
识别纯黑/暗背景视频（如太空、星球类）
"""
import cv2
import os
import json
import numpy as np

OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"

# 亮度判定阈值
DARK_PIXEL_THRESHOLD = 15     # 像素值低于此视为"极暗" (0-255)
DARK_RATIO_THRESHOLD = 0.55   # 超过55%的像素极暗 => 纯黑/暗背景
MEAN_BRIGHTNESS_THRESHOLD = 40  # 平均亮度低于40 => 暗背景
NUM_FRAMES = 5                # 每个视频采样帧数

def analyze_video(filepath):
    """分析单个视频的亮度特征"""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    
    # 均匀采样帧
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    brightness_values = []
    dark_ratios = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 平均亮度
        mean_brightness = float(np.mean(gray))
        brightness_values.append(mean_brightness)
        
        # 暗像素占比
        dark_pixels = np.sum(gray < DARK_PIXEL_THRESHOLD)
        total_pixels = gray.size
        dark_ratio = float(dark_pixels) / total_pixels
        dark_ratios.append(dark_ratio)
    
    cap.release()
    
    if not brightness_values:
        return None
    
    avg_brightness = np.mean(brightness_values)
    avg_dark_ratio = np.mean(dark_ratios)
    max_dark_ratio = np.max(dark_ratios)
    
    return {
        'avg_brightness': round(avg_brightness, 1),
        'avg_dark_ratio': round(avg_dark_ratio, 3),
        'max_dark_ratio': round(max_dark_ratio, 3),
        'is_dark': avg_brightness < MEAN_BRIGHTNESS_THRESHOLD or avg_dark_ratio > DARK_RATIO_THRESHOLD,
    }

def main():
    print("=" * 70)
    print("视频背景亮度分析 - 识别纯黑/暗背景视频")
    print("=" * 70)
    
    # 获取所有 mp4 文件
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')])
    print(f"总视频数: {len(files)}")
    print(f"采样帧数/视频: {NUM_FRAMES}")
    print(f"暗像素阈值: <{DARK_PIXEL_THRESHOLD}/255, 占比>{DARK_RATIO_THRESHOLD*100:.0f}%")
    print(f"平均亮度阈值: <{MEAN_BRIGHTNESS_THRESHOLD}/255")
    print("-" * 70)
    
    results = []
    dark_videos = []
    
    for i, fname in enumerate(files):
        filepath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        result = analyze_video(filepath)
        if result is None:
            print(f"  [{i+1:3d}/{len(files)}] [ERROR] {fname[:55]}")
            continue
        
        result['filename'] = fname
        result['size_mb'] = round(size_mb, 1)
        results.append(result)
        
        if result['is_dark']:
            dark_videos.append(fname)
            status = "[DARK] "
        else:
            status = "[OK]   "
        
        print(f"  [{i+1:3d}/{len(files)}] {status} 亮度={result['avg_brightness']:5.1f} 暗占比={result['avg_dark_ratio']*100:5.1f}% | {fname[:50]}")
    
    print("\n" + "=" * 70)
    print(f"分析完成: 共 {len(results)} 个视频")
    print(f"纯黑/暗背景视频: {len(dark_videos)} 个")
    print("-" * 70)
    
    if dark_videos:
        print("\n待删除的暗背景视频:")
        total_size = 0
        for fname in dark_videos:
            fpath = os.path.join(OUTPUT_DIR, fname)
            size = os.path.getsize(fpath) / (1024 * 1024)
            total_size += size
            print(f"  {fname[:65]:65s} {size:7.1f} MB")
        print(f"\n将释放空间: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    
    # 保存分析结果
    results_path = os.path.join(OUTPUT_DIR, "_brightness_analysis.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存: {results_path}")

if __name__ == "__main__":
    main()
