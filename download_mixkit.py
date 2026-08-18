"""
Mixkit 免费视频批量下载脚本
从 Mixkit 抓取无人机航拍 + 女性户外 + 旋转运镜相关视频并下载到本地
"""
import re
import os
import sys
import json
import time
import requests
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 配置 ============
OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"
BASE_URL = "https://mixkit.co"

# Mixkit 免费视频分类页 - 与"美女户外 + 无人机航拍 + 旋转运镜"相关
CATEGORIES = [
    "https://mixkit.co/free-stock-video/drone/",
    "https://mixkit.co/free-stock-video/drone/?page=2",
    "https://mixkit.co/free-stock-video/drone/?page=3",
    "https://mixkit.co/free-stock-video/woman/",
    "https://mixkit.co/free-stock-video/woman/?page=2",
    "https://mixkit.co/free-stock-video/woman/?page=3",
    "https://mixkit.co/free-stock-video/nature/",
    "https://mixkit.co/free-stock-video/nature/?page=2",
    "https://mixkit.co/free-stock-video/aerial/",
    "https://mixkit.co/free-stock-video/aerial/?page=2",
    "https://mixkit.co/free-stock-video/forest/",
    "https://mixkit.co/free-stock-video/landscape/",
    "https://mixkit.co/free-stock-video/beach/",
    "https://mixkit.co/free-stock-video/mountain/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://mixkit.co/",
}

# 下载视频的 headers（需要 Referer）
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://mixkit.co/",
}

MAX_WORKERS = 4       # 并发下载数
MAX_VIDEOS = 30       # 最多下载视频数
TIMEOUT = 30          # 请求超时秒
CHUNK_SIZE = 1024 * 256  # 256KB chunks

# ============ 逻辑 ============

def fetch_page(url):
    """获取页面 HTML"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [WARN] 获取页面失败 {url}: {e}")
        return None


def extract_video_items(html, page_url):
    """
    从 Mixkit 页面 HTML 中提取视频信息
    Mixkit 页面结构: 每个视频项有 data 属性和 mp4 链接
    返回 list of dict: {url, title, page_link}
    """
    videos = []
    seen = set()

    # 提取所有 mp4 直链（优先 1080p，降级到 720p）
    # Mixkit 有两种 URL 格式:
    #   https://assets.mixkit.co/videos/1012/1012-1080.mp4
    #   https://assets.mixkit.co/active_storage/video_items/100415/1724198576/100415-video-1080.mp4
    mp4_pattern = r'(https://assets\.mixkit\.co/[^"\']+\.mp4)'
    mp4_matches = re.findall(mp4_pattern, html)

    # 提取视频标题/描述用于命名
    # Mixkit 视频项通常有 alt 属性或 data-video-title
    title_pattern = r'data-video-title="([^"]+)"'
    titles = re.findall(title_pattern, html)

    # 另一种标题格式: alt="..."
    alt_pattern = r'<img[^>]*alt="([^"]+)"[^>]*>'
    alts = re.findall(alt_pattern, html)

    # 提取视频详情页链接
    detail_pattern = r'href="(/free-stock-video/[^"]+)"'
    detail_links = re.findall(detail_pattern, html)
    detail_links = [urljoin(BASE_URL, l) for l in detail_links if not l.endswith(('/', 'drone/', 'woman/', 'nature/'))]

    # 去重并收集
    for i, mp4_url in enumerate(mp4_matches):
        # 偏好 1080p，如果没有 720p 也行
        if mp4_url in seen:
            continue

        # 如果有同 id 的 1080 和 720，只取 1080
        # 提取 video id 做去重
        id_match = re.search(r'/(\d+)', mp4_url)
        vid_id = id_match.group(1) if id_match else mp4_url

        if vid_id in seen:
            # 已有该 id 的视频，检查是否 1080 优先
            existing = next((v for v in videos if v['id'] == vid_id), None)
            if existing and '1080' in mp4_url and '1080' not in existing['url']:
                existing['url'] = mp4_url  # 升级到 1080p
            continue

        seen.add(vid_id)

        # 尝试获取标题
        title = ""
        if i < len(titles):
            title = titles[i]
        elif i < len(alts):
            title = alts[i]
        if not title:
            title = f"mixkit_{vid_id}"

        # 清理标题用于文件名
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:60]
        if not safe_title:
            safe_title = f"mixkit_{vid_id}"

        videos.append({
            'id': vid_id,
            'url': mp4_url,
            'title': title,
            'filename': f"{safe_title}_{vid_id}.mp4",
            'page': page_url,
        })

    return videos


def download_video(video, output_dir):
    """下载单个视频"""
    filepath = os.path.join(output_dir, video['filename'])

    # 如果文件已存在且完整，跳过
    if os.path.exists(filepath):
        # 检查文件大小是否合理（>100KB）
        if os.path.getsize(filepath) > 102400:
            print(f"  [SKIP] 已存在: {video['filename']} ({os.path.getsize(filepath)//1024}KB)")
            return True, video

    try:
        r = requests.get(video['url'], headers=DOWNLOAD_HEADERS, timeout=120, stream=True)
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)
        print(f"  [OK] {video['filename']} ({size_mb:.1f} MB)")
        return True, video

    except Exception as e:
        print(f"  [FAIL] {video['filename']}: {e}")
        # 删除不完整的文件
        if os.path.exists(filepath):
            os.remove(filepath)
        return False, video


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Mixkit 免费视频批量下载")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"最大下载数: {MAX_VIDEOS}")
    print("=" * 60)

    # 第一步: 抓取所有分类页，收集视频链接
    all_videos = []
    seen_ids = set()

    print("\n[1/3] 抓取视频列表...")
    for cat_url in CATEGORIES:
        print(f"  抓取: {cat_url}")
        html = fetch_page(cat_url)
        if not html:
            continue
        videos = extract_video_items(html, cat_url)
        for v in videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                all_videos.append(v)
        print(f"    发现 {len(videos)} 个视频，累计 {len(all_videos)} 个")
        time.sleep(0.5)  # 礼貌延迟

    print(f"\n总共发现 {len(all_videos)} 个不重复视频")

    if not all_videos:
        print("[ERROR] 没有找到任何视频，请检查网络")
        sys.exit(1)

    # 保存视频信息列表
    info_path = os.path.join(OUTPUT_DIR, "_video_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    print(f"视频信息已保存: {info_path}")

    # 第二步: 下载视频
    to_download = all_videos[:MAX_VIDEOS]
    print(f"\n[2/3] 开始下载 {len(to_download)} 个视频 (并发: {MAX_WORKERS})...")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_video, v, OUTPUT_DIR): v for v in to_download}
        for i, future in enumerate(as_completed(futures)):
            ok, video = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            print(f"  进度: {i+1}/{len(to_download)} (成功: {success}, 失败: {failed})")

    # 第三步: 汇总
    print(f"\n[3/3] 下载完成")
    print("=" * 60)
    print(f"成功: {success} 个")
    print(f"失败: {failed} 个")
    print(f"输出目录: {OUTPUT_DIR}")

    # 列出已下载文件
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
    print(f"已下载文件: {len(files)} 个, 总大小: {total_size/(1024*1024):.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
