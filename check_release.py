import urllib.request
import json
import os

url = 'https://api.github.com/repos/usrname0/YaGUFF/releases'
req = urllib.request.Request(url)
req.add_header('Accept', 'application/vnd.github.v3+json')

github_token = os.environ.get('GITHUB_TOKEN')
if github_token:
    req.add_header('Authorization', f'token {github_token}')

try:
    response = urllib.request.urlopen(req)
    releases = json.loads(response.read().decode())

    print(f"Total releases: {len(releases)}")
    print("\nRecent releases:")
    for release in releases[:5]:
        tag = release.get('tag_name', 'No tag')
        published = release.get('published_at', 'Unknown')
        draft = ' (DRAFT)' if release.get('draft') else ''
        print(f"  {tag} - {published}{draft}")
except Exception as e:
    print(f"Error: {e}")
