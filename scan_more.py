"""
抓取 Mixkit 更多分类页，筛选航拍视角+人物户外视频
"""
import re
import os
import json
import time
import requests
from urllib.parse import urljoin

BASE_URL = "https://mixkit.co"
OUTPUT_DIR = r"D:\Python Project\flash3d-main\pretrain_dataset"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://mixkit.co/",
}

# 新增分类页 - 人物/户外/旅行相关
NEW_CATEGORIES = [
    "https://mixkit.co/free-stock-video/people/",
    "https://mixkit.co/free-stock-video/people/?page=2",
    "https://mixkit.co/free-stock-video/people/?page=3",
    "https://mixkit.co/free-stock-video/lifestyle/",
    "https://mixkit.co/free-stock-video/lifestyle/?page=2",
    "https://mixkit.co/free-stock-video/lifestyle/?page=3",
    "https://mixkit.co/free-stock-video/travel/",
    "https://mixkit.co/free-stock-video/travel/?page=2",
    "https://mixkit.co/free-stock-video/travel/?page=3",
    "https://mixkit.co/free-stock-video/beach/",
    "https://mixkit.co/free-stock-video/beach/?page=2",
    "https://mixkit.co/free-stock-video/beach/?page=3",
    "https://mixkit.co/free-stock-video/swim/",
    "https://mixkit.co/free-stock-video/swim/?page=2",
    "https://mixkit.co/free-stock-video/fashion/",
    "https://mixkit.co/free-stock-video/fashion/?page=2",
    "https://mixkit.co/free-stock-video/fitness/",
    "https://mixkit.co/free-stock-video/fitness/?page=2",
    "https://mixkit.co/free-stock-video/active/",
    "https://mixkit.co/free-stock-video/active/?page=2",
]

# 航拍/俯视关键词 - 这是核心筛选条件
AERIAL_KEYWORDS = [
    'from above', 'seen from above', 'overhead', 'top view', "bird's eye",
    'aerial', 'drone', 'looking down', 'from the air', 'top-down',
    'spinning', 'orbit', 'rotation', 'pan', 'zoom', 'pull back', 'pullback',
    'push in', 'dolly', 'circling', 'circl', 'rotating', 'rotat'
]

# 人物关键词
PERSON_KEYWORDS = [
    'woman', 'girl', 'female', 'lady', 'couple', 'people', 'person',
    'beach', 'walking', 'running', 'swimming', 'yoga', 'dancing', 'surfer',
    'bather', 'sunbath', 'model'
]

def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [WARN] {url}: {e}")
        return None

def extract_videos(html, page_url):
    videos = []
    seen = set()

    mp4_pattern = r'(https://assets\.mixkit\.co/[^"\']+\.mp4)'
    mp4_matches = re.findall(mp4_pattern, html)

    title_pattern = r'data-video-title="([^"]+)"'
    titles = re.findall(title_pattern, html)

    alt_pattern = r'<img[^>]*alt="([^"]+)"[^>]*>'
    alts = re.findall(alt_pattern, html)

    for i, mp4_url in enumerate(mp4_matches):
        id_match = re.search(r'/(\d+)', mp4_url)
        vid_id = id_match.group(1) if id_match else mp4_url
        if vid_id in seen:
            existing = next((v for v in videos if v['id'] == vid_id), None)
            if existing and '1080' in mp4_url and '1080' not in existing['url']:
                existing['url'] = mp4_url
            continue
        seen.add(vid_id)

        title = ""
        if i < len(titles):
            title = titles[i]
        elif i < len(alts):
            title = alts[i]
        if not title:
            title = f"mixkit_{vid_id}"

        videos.append({
            'id': vid_id,
            'url': mp4_url,
            'title': title,
            'page': page_url,
        })

    return videos

def main():
    print("=" * 60)
    print("抓取更多 Mixkit 分类，筛选航拍视角+人物视频")
    print("=" * 60)

    all_new = []
    seen_ids = set()

    # 加载已有视频 ID，避免重复
    info_path = os.path.join(OUTPUT_DIR, "_video_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        seen_ids = {v['id'] for v in existing}
        print(f"已有视频 ID: {len(seen_ids)} 个")

    print("\n[1/2] 抓取新分类页...")
    for cat_url in NEW_CATEGORIES:
        html = fetch_page(cat_url)
        if not html:
            continue
        videos = extract_videos(html, cat_url)
        new_count = 0
        for v in videos:
            if v['id'] not in seen_ids:
                seen_ids.add(v['id'])
                all_new.append(v)
                new_count += 1
        print(f"  {cat_url.split('/free-stock-video/')[1][:20]:20s} -> 新增 {new_count} 个 (累计 {len(all_new)})")
        time.sleep(0.3)

    print(f"\n新发现视频: {len(all_new)} 个")

    # 筛选航拍视角+人物的视频
    print("\n[2/2] 筛选航拍视角+人物视频...")
    aerial_person = []
    aerial_only = []
    person_outdoor = []

    for v in all_new:
        title_lower = v['title'].lower()
        has_aerial = any(kw in title_lower for kw in AERIAL_KEYWORDS)
        has_person = any(kw in title_lower for kw in PERSON_KEYWORDS)

        if has_aerial and has_person:
            aerial_person.append(v)
        elif has_aerial:
            aerial_only.append(v)
        elif has_person:
            person_outdoor.append(v)

    print(f"\n=== 航拍视角+人物 (最匹配): {len(aerial_person)} ===")
    for v in aerial_person:
        res = '1080p' if '1080' in v['url'] else '360p'
        print(f"  [{res}] ID:{v['id']:6s} | {v['title']}")

    print(f"\n=== 纯航拍视角 (无人像): {len(aerial_only)} ===")
    for v in aerial_only[:15]:
        res = '1080p' if '1080' in v['url'] else '360p'
        print(f"  [{res}] ID:{v['id']:6s} | {v['title']}")

    print(f"\n=== 人物户外 (非航拍): {len(person_outdoor)} ===")
    for v in person_outdoor[:15]:
        res = '1080p' if '1080' in v['url'] else '360p'
        print(f"  [{res}] ID:{v['id']:6s} | {v['title']}")

    # 保存所有新视频信息
    all_new_path = os.path.join(OUTPUT_DIR, "_new_videos.json")
    with open(all_new_path, 'w', encoding='utf-8') as f:
        json.dump(all_new, f, ensure_ascii=False, indent=2)
    print(f"\n新视频信息已保存: {all_new_path}")

    # 保存筛选结果
    result = {
        'aerial_person': aerial_person,
        'aerial_only': aerial_only,
        'person_outdoor': person_outdoor,
    }
    result_path = os.path.join(OUTPUT_DIR, "_filtered_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"筛选结果已保存: {result_path}")

if __name__ == "__main__":
    main()
