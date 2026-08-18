"""诊断 Pexels 搜索和 JSON 提取问题"""
from curl_cffi import requests
import re
import json

# 测试1: 搜索是否还能工作
print("=== 测试1: 搜索可达性 ===")
try:
    r = requests.get('https://www.pexels.com/search/videos/aerial%20woman%20outdoor/', 
                     impersonate='chrome', timeout=30)
    print(f"状态码: {r.status_code}")
    print(f"内容长度: {len(r.text)}")
except Exception as e:
    print(f"错误: {e}")
    import sys; sys.exit(1)

html = r.text

# 测试2: __NEXT_DATA__ 是否存在
print("\n=== 测试2: __NEXT_DATA__ ===")
json_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
if json_match:
    print(f"找到, 长度: {len(json_match.group(1))}")
    data = json.loads(json_match.group(1))
    
    # 导航到视频数据
    videos_data = data.get('props', {}).get('pageProps', {}).get('initialData', {}).get('data', [])
    print(f"视频数: {len(videos_data)}")
    
    if videos_data:
        v = videos_data[0]
        attrs = v.get('attributes', {})
        print(f"attributes keys: {list(attrs.keys())}")
        
        # 检查 video_files 在哪
        print(f"\n'video_files' in attrs: {'video_files' in attrs}")
        print(f"'video' in attrs: {'video' in attrs}")
        
        if 'video' in attrs:
            video_field = attrs['video']
            print(f"video field type: {type(video_field)}")
            if isinstance(video_field, dict):
                print(f"video field keys: {list(video_field.keys())}")
                # 可能有 video_files 在这里
                if 'video_files' in video_field:
                    vf = video_field['video_files']
                    print(f"video_files: {len(vf)} 个")
                    if vf:
                        print(f"第一个: {json.dumps(vf[0], indent=2)[:200]}")
        
        # 也直接检查 video_files
        if 'video_files' in attrs:
            vf = attrs['video_files']
            print(f"\n直接 video_files: {len(vf)} 个")
            if vf:
                print(f"第一个: {json.dumps(vf[0], indent=2)[:200]}")
        
        # 检查 video_files 可能在顶层
        if 'video_files' in v:
            vf = v['video_files']
            print(f"\n顶层 video_files: {len(vf)} 个")
else:
    print("未找到 __NEXT_DATA__")
    # 检查是否有 Cloudflare challenge
    if 'cloudflare' in html.lower() or 'challenge' in html.lower():
        print("检测到 Cloudflare challenge!")
    print(f"HTML 前500字符: {html[:500]}")
