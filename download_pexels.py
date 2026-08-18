"""
Pexels 航拍视角美女户外视频下载脚本
使用 curl_cffi 绕过 Cloudflare 反爬虫保护
"""
import re
import os
import sys
import json
import time
import requests
from curl_cffi import requests as cffi_requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"

# Pexels 搜索 URL 模板
def search_url(query, page=1):
    return f"https://www.pexels.com/search/videos/{quote(query)}/?page={page}"

# 搜索关键词组合 - 航拍视角 + 美女 + 户外
SEARCH_QUERIES = [
    "aerial woman outdoor",
    "drone woman beach",
    "aerial girl nature",
    "woman from above",
    "drone shot woman",
    "aerial female beach",
    "drone aerial woman",
    "top view woman outdoor",
    "bird eye view woman",
    "aerial woman swimming",
    "drone woman walking",
    "aerial woman sunset",
    "orbit shot woman",
    "aerial woman field",
    "drone woman mountain",
]

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# 下载用普通 requests + 浏览器 headers（Pexels CDN 可能不需要 cffi）
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.pexels.com/",
}

MAX_WORKERS = 4
MAX_VIDEOS = 40
TIMEOUT = 30

def search_pexels(query, page=1):
    """用 curl_cffi 搜索 Pexels 视频，返回视频信息列表"""
    url = search_url(query, page)
    try:
        r = cffi_requests.get(url, impersonate='chrome', timeout=TIMEOUT, headers=BROWSER_HEADERS)
        if r.status_code != 200:
            print(f"  [WARN] {query} page{page}: HTTP {r.status_code}")
            return []
        
        html = r.text
        videos = []
        
        # Pexels 页面中视频数据通常在 JSON 嵌入的 script 标签或 data 属性中
        # 提取视频文件链接: https://videos.pexels.com/video-files/ID/ID-hd_1920_1080_30fps.mp4
        # 每个视频有多个分辨率，优先取 HD 1080p
        
        # 方法1: 从 JSON 数据提取
        # Pexels 搜索页有 __NEXT_DATA__ 或类似 JSON
        json_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                # 解析视频数据
                search_results = data.get('props', {}).get('pageProps', {}).get('searchResults', {})
                media = search_results.get('data', [])
                for item in media:
                    if item.get('type') != 'Video':
                        continue
                    vid_id = str(item.get('id', ''))
                    # 获取视频文件
                    video_files = item.get('video_files', [])
                    # 优先 HD 1080p
                    best_url = None
                    best_height = 0
                    for vf in video_files:
                        if vf.get('file_type') == 'video/mp4':
                            h = vf.get('height', 0)
                            if h > best_height and h <= 1080:
                                best_height = h
                                best_url = vf.get('link')
                    if not best_url and video_files:
                        best_url = video_files[0].get('link')
                    
                    if best_url:
                        videos.append({
                            'id': vid_id,
                            'url': best_url,
                            'title': item.get('alt', item.get('description', f'pexels_{vid_id}')),
                            'width': item.get('width'),
                            'height': best_height,
                            'duration': item.get('duration'),
                            'query': query,
                        })
            except json.JSONDecodeError:
                pass
        
        # 方法2: 如果 JSON 解析失败，用正则提取
        if not videos:
            # 提取所有唯一视频 ID 和对应 mp4
            mp4_pattern = r'https://videos\.pexels\.com/video-files/(\d+)/(\d+)-(hd_\d+_\d+|sd_\d+_\d+)[^"]*\.mp4'
            matches = re.findall(mp4_pattern, html)
            seen_ids = {}
            for file_id, vid_id, quality in matches:
                if vid_id not in seen_ids or 'hd' in quality:
                    full_match = re.search(rf'https://videos\.pexels\.com/video-files/{file_id}/{file_id}-{quality}[^"]*\.mp4', html)
                    if full_match:
                        seen_ids[vid_id] = full_match.group(0)
            
            for vid_id, url in seen_ids.items():
                videos.append({
                    'id': vid_id,
                    'url': url,
                    'title': f'pexels_{vid_id}',
                    'height': 1080 if 'hd_1920_1080' in url else (720 if '720' in url else 360),
                    'query': query,
                })
        
        return videos
    except Exception as e:
        print(f"  [ERROR] {query} page{page}: {e}")
        return []

def download_video(video, output_dir):
    """下载单个视频"""
    safe_title = re.sub(r'[^\w\s-]', '', video['title']).strip().replace(' ', '_')[:50]
    if not safe_title:
        safe_title = f"pexels_{video['id']}"
    filename = f"{safe_title}_{video['id']}.mp4"
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 102400:
        print(f"  [SKIP] {filename} ({os.path.getsize(filepath)//(1024*1024)}MB)")
        return True
    
    try:
        r = requests.get(video['url'], headers=DOWNLOAD_HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*256):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        size_mb = downloaded / (1024*1024)
        print(f"  [OK] {filename} ({size_mb:.1f}MB, {video.get('height','?')}p)")
        return True
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Pexels 航拍视角美女户外视频下载")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 搜索阶段
    all_videos = []
    seen_ids = set()
    
    # 加载已下载视频 ID（避免与 Mixkit 重复）
    info_path = os.path.join(OUTPUT_DIR, "_video_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        seen_ids = {v['id'] for v in existing}
    
    print(f"\n[1/3] 搜索 Pexels 视频...")
    for query in SEARCH_QUERIES:
        print(f"  搜索: '{query}'")
        videos = search_pexels(query, page=1)
        new_count = 0
        for v in videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                all_videos.append(v)
                new_count += 1
        print(f"    -> 发现 {len(videos)} 个, 新增 {new_count} 个 (累计 {len(all_videos)})")
        time.sleep(1)  # 礼貌延迟
    
    print(f"\n总共发现 {len(all_videos)} 个不重复视频")
    
    if not all_videos:
        print("[ERROR] 没有找到视频")
        sys.exit(1)
    
    # 保存搜索结果
    results_path = os.path.join(OUTPUT_DIR, "_pexels_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    print(f"搜索结果已保存: {results_path}")
    
    # 打印前20个视频信息
    print(f"\n视频列表 (前20个):")
    for v in all_videos[:20]:
        h = v.get('height', '?')
        print(f"  [{h}p] ID:{v['id']:8s} | {v['title'][:50]}")
    
    # 下载阶段
    to_download = all_videos[:MAX_VIDEOS]
    print(f"\n[2/3] 开始下载 {len(to_download)} 个视频 (并发: {MAX_WORKERS})...")
    print("-" * 60)
    
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_video, v, OUTPUT_DIR): v for v in to_download}
        for i, future in enumerate(as_completed(futures)):
            ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            print(f"  >>> 进度: {i+1}/{len(to_download)} (成功:{success} 失败:{failed})")
    
    # 汇总
    print(f"\n[3/3] 下载完成")
    print("=" * 60)
    print(f"成功: {success}, 失败: {failed}")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
    print(f"目录总文件: {len(files)} 个, 总大小: {total_size/(1024*1024*1024):.2f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()
