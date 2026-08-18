"""
Pexels 航拍视角美女户外视频 - 完整版下载脚本
正确提取视频标题，下载全部视频并重命名已有文件
"""
import re
import os
import sys
import json
import time
import requests
from curl_cffi import requests as cffi_requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"

SEARCH_QUERIES = [
    "aerial woman outdoor",
    "drone woman beach",
    "aerial girl nature",
    "woman from above",
    "drone shot woman",
    "aerial female beach",
    "top view woman outdoor",
    "aerial woman swimming",
    "aerial woman sunset",
    "orbit shot woman",
    "aerial woman field",
    "drone woman mountain",
    "aerial woman walking",
    "drone woman forest",
    "aerial woman running",
]

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.pexels.com/",
}

MAX_WORKERS = 5
TIMEOUT = 30

def search_pexels(query, page=1):
    """搜索 Pexels 视频，正确提取标题和视频文件"""
    url = f"https://www.pexels.com/search/videos/{quote(query)}/?page={page}"
    try:
        r = cffi_requests.get(url, impersonate='chrome', timeout=TIMEOUT, headers=BROWSER_HEADERS)
        if r.status_code != 200:
            return []
        
        html = r.text
        json_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL)
        if not json_match:
            return []
        
        data = json.loads(json_match.group(1))
        videos_data = data.get('props', {}).get('pageProps', {}).get('initialData', {}).get('data', [])
        
        videos = []
        for item in videos_data:
            if item.get('type') not in ('Video', 'video'):
                continue
            attrs = item.get('attributes', {})
            vid_id = str(item.get('id', ''))
            
            # 获取最佳视频文件 (优先 1080p)
            # video_files 在 attributes.video.video_files 中
            video_data = attrs.get('video', {})
            video_files = video_data.get('video_files', [])
            best_url = None
            best_height = 0
            for vf in video_files:
                if vf.get('file_type') == 'video/mp4':
                    h = vf.get('height', 0)
                    if h > best_height and h <= 1080:
                        best_height = h
                        best_url = vf.get('link')
            if not best_url and video_files:
                for vf in video_files:
                    if vf.get('file_type') == 'video/mp4':
                        best_url = vf.get('link')
                        best_height = vf.get('height', 0)
                        break
            
            if not best_url:
                continue
            
            description = attrs.get('description', '')
            title = attrs.get('title', '')
            display_title = description if description else title
            
            videos.append({
                'id': vid_id,
                'url': best_url,
                'title': display_title,
                'alt_title': title,
                'height': best_height,
                'duration': attrs.get('duration'),
                'query': query,
            })
        
        return videos
    except Exception as e:
        print(f"  [ERROR] {query} p{page}: {e}")
        return []

def safe_filename(title, vid_id):
    """生成安全文件名"""
    if not title:
        title = f"pexels_{vid_id}"
    safe = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:60]
    if not safe:
        safe = f"pexels_{vid_id}"
    return f"{safe}_{vid_id}.mp4"

def download_video(video, output_dir):
    """下载单个视频"""
    filename = safe_filename(video['title'], video['id'])
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 102400:
        return True, filename, 'skip'
    
    try:
        r = requests.get(video['url'], headers=DOWNLOAD_HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*256):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        return True, filename, 'ok'
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, filename, str(e)[:50]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Pexels 航拍视角美女户外视频 - 完整下载")
    print("=" * 70)
    
    # 搜索阶段
    all_videos = []
    seen_ids = set()
    
    print("\n[1/3] 搜索 Pexels 视频...")
    for query in SEARCH_QUERIES:
        print(f"  搜索: '{query}'", end='', flush=True)
        videos = search_pexels(query, page=1)
        new_count = 0
        for v in videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                all_videos.append(v)
                new_count += 1
        print(f" -> {len(videos)}个, 新增{new_count} (累计{len(all_videos)})")
        time.sleep(1)
    
    print(f"\n总共发现 {len(all_videos)} 个不重复视频")
    
    # 保存完整搜索结果（带标题）
    results_path = os.path.join(OUTPUT_DIR, "_pexels_full_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    
    # 打印视频列表
    print(f"\n视频列表:")
    for v in all_videos:
        h = v.get('height', '?')
        dur = v.get('duration', '?')
        print(f"  [{h}p {dur}s] ID:{v['id']:8s} | {v['title'][:55]}")
    
    # 下载阶段 - 下载全部
    print(f"\n[2/3] 下载 {len(all_videos)} 个视频 (并发: {MAX_WORKERS})...")
    print("-" * 70)
    
    success = 0
    failed = 0
    skipped = 0
    failed_list = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_video, v, OUTPUT_DIR): v for v in all_videos}
        for i, future in enumerate(as_completed(futures)):
            ok, filename, status = future.result()
            if ok:
                success += 1
                if status == 'skip':
                    skipped += 1
            else:
                failed += 1
                failed_list.append(filename)
            if (i+1) % 10 == 0 or i+1 == len(all_videos):
                print(f"  进度: {i+1}/{len(all_videos)} (成功:{success} 跳过:{skipped} 失败:{failed})")
    
    # 清理旧的 pexels_数字 格式文件（已被正确命名的替代）
    print(f"\n[3/3] 清理旧文件...")
    old_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('pexels_') and re.match(r'^pexels_\d+_\d+\.mp4$', f)]
    for old in old_files:
        old_path = os.path.join(OUTPUT_DIR, old)
        os.remove(old_path)
    print(f"  清理旧命名文件: {len(old_files)} 个")
    
    # 汇总
    print("\n" + "=" * 70)
    print(f"下载完成!")
    print(f"  成功下载: {success - skipped} 个")
    print(f"  跳过(已存在): {skipped} 个")
    print(f"  失败: {failed} 个")
    if failed_list:
        print(f"  失败列表: {failed_list[:5]}")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
    print(f"\n目录总文件: {len(files)} 个, 总大小: {total_size/(1024*1024*1024):.2f} GB")
    print("=" * 70)

if __name__ == "__main__":
    main()
