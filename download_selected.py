"""
Mixkit 精准下载脚本 - 筛选美女户外+旋转运镜/航拍视频
从已抓取的 _video_info.json 中筛选最相关视频，尝试升级到 1080p 后下载
"""
import re
import os
import sys
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"
INFO_PATH = os.path.join(OUTPUT_DIR, "_video_info.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://mixkit.co/",
}

MAX_WORKERS = 3
TIMEOUT = 60
CHUNK_SIZE = 1024 * 256

# ============ 筛选关键词 ============
# 户外场景关键词
OUTDOOR_KEYWORDS = [
    'sunset', 'beach', 'park', 'field', 'nature', 'outdoor', 'sky', 'sun',
    'mountain', 'lake', 'pool', 'swimming', 'walk', 'walking', 'yoga',
    'spinning', 'spins', 'circle', 'dance', 'dancing', 'run', 'running',
    'sports car', 'bike', 'meditat', 'cloud', 'flower', 'plain',
    'rippling', 'reflection', 'seen from above', 'aerial', 'drone'
]
# 人物关键词
PERSON_KEYWORDS = ['woman', 'girl', 'female', 'couple', 'lover', 'friend', 'people']

# 排除关键词（室内/不相关场景）
EXCLUDE_KEYWORDS = [
    'office', 'nightclub', 'club', 'homework', 'library', 'microscope',
    'reporter', 'green screen', 'chroma', 'staircase', 'floor', 'ghost',
    'laboratory', 'scientist', 'newsroom', 'strength training',
    'yogurt', 'bowl', 'fruit', 'homework', 'smoking', 'crying',
    'lamenting', 'desperate', 'heartbroken', 'upset', 'sad', 'screaming',
    'abandoned', 'night', 'dark', 'smoke'
]

def matches_keywords(title):
    """检查标题是否符合美女户外+运镜要求"""
    title_lower = title.lower()
    
    # 排除不相关内容
    for kw in EXCLUDE_KEYWORDS:
        if kw in title_lower:
            return False, "excluded"
    
    # 必须包含人物关键词
    has_person = any(kw in title_lower for kw in PERSON_KEYWORDS)
    if not has_person:
        return False, "no_person"
    
    # 检查户外/运镜关键词
    matched_outdoor = [kw for kw in OUTDOOR_KEYWORDS if kw in title_lower]
    if len(matched_outdoor) == 0:
        return False, "no_outdoor"
    
    return True, matched_outdoor

def upgrade_resolution(url):
    """尝试将视频 URL 从 360p 升级到 1080p"""
    if '-360.mp4' in url:
        url_1080 = url.replace('-360.mp4', '-1080.mp4')
        try:
            r = requests.head(url_1080, headers=HEADERS, timeout=15, allow_redirects=True)
            if r.status_code == 200 and 'video' in r.headers.get('content-type', ''):
                size = int(r.headers.get('content-length', 0))
                if size > 102400:  # >100KB 才算有效
                    return url_1080, size
        except:
            pass
    return url, None

def download_video(video, output_dir):
    """下载单个视频，先尝试 1080p"""
    filepath = os.path.join(output_dir, video['filename'])
    
    # 已存在则跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 102400:
        print(f"  [SKIP] {video['filename']} ({os.path.getsize(filepath)//1024}KB)")
        return True
    
    url = video['url']
    # 尝试升级分辨率
    if '-360.mp4' in url:
        upgraded, size = upgrade_resolution(url)
        if upgraded != url:
            print(f"  [UPGRADE] {video['id']}: 360p -> 1080p ({size//(1024*1024)}MB)")
            url = upgraded
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        size_mb = downloaded / (1024 * 1024)
        print(f"  [OK] {video['filename']} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {video['filename']}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载已抓取的视频信息
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
        all_videos = json.load(f)
    
    print("=" * 60)
    print("精准筛选: 美女户外 + 旋转运镜/航拍视频")
    print("=" * 60)
    
    # 筛选符合条件的美女户外视频
    selected = []
    for v in all_videos:
        ok, reason = matches_keywords(v['title'])
        if ok:
            # 清理文件名
            safe_title = re.sub(r'[^\w\s-]', '', v['title']).strip().replace(' ', '_')[:60]
            if not safe_title:
                safe_title = f"mixkit_{v['id']}"
            v['filename'] = f"{safe_title}_{v['id']}.mp4"
            v['matched_keywords'] = reason
            selected.append(v)
    
    print(f"\n筛选结果: {len(selected)} 个符合「美女户外+运镜」的视频")
    print("-" * 60)
    for v in selected:
        res = '1080p' if '1080' in v['url'] else '360p'
        print(f"  [{res:5s}] ID:{v['id']:6s} | {v['title'][:55]}")
        print(f"          关键词: {', '.join(v['matched_keywords'][:5])}")
    
    if not selected:
        print("\n[ERROR] 没有筛选到符合条件的视频")
        sys.exit(1)
    
    print(f"\n开始下载 {len(selected)} 个视频 (并发: {MAX_WORKERS})...")
    print("-" * 60)
    
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_video, v, OUTPUT_DIR): v for v in selected}
        for i, future in enumerate(as_completed(futures)):
            ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            print(f"  >>> 进度: {i+1}/{len(selected)} (成功:{success} 失败:{failed})")
    
    print("\n" + "=" * 60)
    print(f"下载完成: 成功 {success}, 失败 {failed}")
    
    # 统计所有已下载文件
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
    print(f"目录总文件: {len(files)} 个, 总大小: {total_size/(1024*1024):.1f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
