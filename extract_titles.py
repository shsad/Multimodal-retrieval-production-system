import json
from pathlib import Path

blog_dir = Path("data/orig_data/blogposts")

blogs = []

for p in blog_dir.glob("*.json"):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    blogs.append((p.stem, data.get("title", "")))

for slug, title in sorted(blogs):
    print(slug, "->", title)