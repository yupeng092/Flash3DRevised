from curl_cffi import requests
import re

try:
    r = requests.get('https://www.pexels.com/search/videos/aerial%20woman%20outdoor/', 
                     impersonate='chrome', timeout=30)
    print(f'状态码: {r.status_code}')
    print(f'内容长度: {len(r.text)}')
    
    mp4s = re.findall(r'(https://videos\.pexels\.com/[^"\s]+\.mp4[^"\s]*)', r.text)
    print(f'mp4链接数: {len(mp4s)}')
    for m in mp4s[:5]:
        print(f'  示例: {m}')
    
    title = re.search(r'<title>([^<]+)</title>', r.text)
    if title:
        print(f'页面标题: {title.group(1)}')
    
    # 查找视频项的数据
    video_ids = re.findall(r'"id":(\d+).*?"video_files"', r.text)
    print(f'视频ID数: {len(video_ids)}')
    
except Exception as e:
    print(f'错误: {e}')
