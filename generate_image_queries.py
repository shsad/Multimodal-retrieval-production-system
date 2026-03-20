import json

with open("data/orig_data/img_to_blog.json") as f:
    mapping = json.load(f)

for img, blog in list(mapping.items())[:20]:
    print(img, "->", blog)