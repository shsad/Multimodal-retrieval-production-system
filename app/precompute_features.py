"""Precompute text and image features for ML6 blog retrieval."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def find_existing_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def split_text_into_chunks(blog_json: dict) -> list[str]:
    """
    Build multiple retrieval chunks per blog post.
    Goal: allow matching against the most relevant section instead of one giant embedding.
    """
    chunks = []

    title = str(blog_json.get("title", "")).strip()
    summary = str(blog_json.get("summary", "")).strip()
    subtitle = str(blog_json.get("subtitle", "")).strip()

    header_chunk_parts = [x for x in [title, subtitle, summary] if x]
    if header_chunk_parts:
        chunks.append("\n".join(header_chunk_parts))

    current_section = []
    for block in blog_json.get("blocks", []):
        if not isinstance(block, dict):
            continue

        block_parts = []
        for key in ["title", "heading", "subheading", "content", "caption", "alt"]:
            value = block.get(key, "")
            if isinstance(value, str) and value.strip():
                block_parts.append(value.strip())

        if not block_parts:
            continue

        block_text = "\n".join(block_parts)

        if any(block.get(k) for k in ["title", "heading", "subheading"]):
            if current_section:
                chunks.append("\n".join(current_section))
                current_section = []
            current_section.append(block_text)
        else:
            current_section.append(block_text)

        joined = "\n".join(current_section)
        if len(joined) > 800:
            chunks.append(joined)
            current_section = []

    if current_section:
        chunks.append("\n".join(current_section))

    if not chunks:
        fallback = "\n".join(header_chunk_parts) if header_chunk_parts else title
        if fallback:
            chunks = [fallback]

    deduped = []
    seen = set()
    for chunk in chunks:
        c = chunk.strip()
        if c and c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped


def encode_clip_image_variants(model, preprocess, image_path: Path) -> np.ndarray:
    """
    Return multiple embeddings per image, one for each transformed variant.
    Shape: [num_variants, dim]
    """
    image = Image.open(image_path).convert("RGB")
    gray = image.convert("L").convert("RGB")

    variants = [
        image,
        gray,
        image.rotate(90, expand=True),
        image.rotate(180, expand=True),
        image.rotate(270, expand=True),
        gray.rotate(90, expand=True),
        gray.rotate(180, expand=True),
        gray.rotate(270, expand=True),
    ]

    embs = []
    with torch.no_grad():
        for img in variants:
            image_input = preprocess(img).unsqueeze(0)
            emb = model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu().numpy()[0].astype(np.float32))

    return np.stack(embs, axis=0).astype(np.float32)


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    blogposts_dir = find_existing_path([
        project_root / "data" / "orig_data" / "blogposts",
        Path("data/orig_data/blogposts"),
    ])

    img_to_blog_path = find_existing_path([
        project_root / "data" / "orig_data" / "img_to_blog.json",
        Path("data/orig_data/img_to_blog.json"),
    ])

    images_dir = find_existing_path([
        project_root / "data" / "orig_data" / "images",
        project_root / "data" / "orig_data" / "blogposts" / "images",
        Path("data/orig_data/images"),
        Path("data/orig_data/blogposts/images"),
    ])

    if blogposts_dir is None:
        raise FileNotFoundError("Could not find blogposts directory.")
    if img_to_blog_path is None:
        raise FileNotFoundError("Could not find img_to_blog.json.")
    if images_dir is None:
        raise FileNotFoundError("Could not find images directory.")

    out_text = script_dir / "text_features.pkl"
    out_image = script_dir / "image_features.pkl"

    print(f"Using blogposts_dir: {blogposts_dir}")
    print(f"Using img_to_blog_path: {img_to_blog_path}")
    print(f"Using images_dir: {images_dir}")

    # Must match the model used in `app/main.py`

    #text_model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    clip_model, preprocess = clip.load("ViT-B/16", device="cpu")

    blog_files = sorted(blogposts_dir.glob("*.json"))
    print(f"Found {len(blog_files)} blog json files")

    # Text features by slug -> stacked chunk embeddings
    text_features = {}

    for blog_file in blog_files:
        with open(blog_file, "r", encoding="utf-8") as f:
            blog = json.load(f)

        slug = blog_file.stem.rstrip("-").strip()
        chunks = split_text_into_chunks(blog)
        if not chunks:
            chunks = [slug]

        embs = text_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if embs.ndim == 1:
            embs = embs[None, :]

        text_features[slug] = embs

    # Image features by image filename -> stacked variant embeddings
    with open(img_to_blog_path, "r", encoding="utf-8") as f:
        img_to_blog = json.load(f)

    image_features = {}

    for image_name, mapped_value in img_to_blog.items():
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        if not isinstance(mapped_value, str):
            continue

        try:
            embs = encode_clip_image_variants(clip_model, preprocess, image_path)
            image_features[image_name] = embs
        except Exception as e:
            print(f"Skipping image {image_name}: {e}")

    with open(out_text, "wb") as f:
        pickle.dump(text_features, f)

    with open(out_image, "wb") as f:
        pickle.dump(image_features, f)

    print(f"Saved text features to: {out_text}")
    print(f"Saved image features to: {out_image}")
    print(f"Text feature count: {len(text_features)}")
    print(f"Image feature count: {len(image_features)}")


if __name__ == "__main__":
    main()