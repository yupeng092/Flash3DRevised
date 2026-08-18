"""检查 Pexels 视频 attributes 结构"""
from curl_cffi import requests
import re
import json

r = requests.get('https://www.pexels.com/search/videos/aerial%20woman%20outdoor/', 
                 impersonate='chrome', timeout=30)

html = r.text
json_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL)
data = json.loads(json_match.group(1))

videos = data['props']['pageProps']['initialData']['data']
print(f"视频数: {len(videos)}")
print()

# 检查第一个视频的完整结构
v = videos[0]
print(f"顶层 keys: {list(v.keys())}")
print(f"  id: {v['id']}")
print(f"  type: {v['type']}")

attrs = v.get('attributes', {})
print(f"  attributes keys: {list(attrs.keys())}")
print()

# 打印标题相关字段
for key in ['alt', 'description', 'title', 'name', 'slug', 'url', 'image', 'duration', 'height', 'width']:
    if key in attrs:
        val = attrs[key]
        if isinstance(val, str) and len(val) > 100:
            val = val[:100] + '...'
        print(f"  {key}: {val}")

# 检查 video_files
if 'video_files' in attrs:
    vf = attrs['video_files']
    print(f"\n  video_files: {len(vf)} 个")
    for f in vf[:3]:
        print(f"    {f.get('quality', '?')} {f.get('width','?')}x{f.get('height','?')} -> {f.get('link','')[:80]}")

# 打印前5个视频的标题
print("\n=== 前5个视频标题 ===")
for v in videos[:5]:
    attrs = v.get('attributes', {})
    title = attrs.get('alt', attrs.get('description', 'N/A'))
    vid_id = v['id']
    print(f"  ID:{vid_id} | {title}")
