"""Module used to serve the ML6 blog post retrieval engine."""

import os
import logging

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError

# Added
import pickle 
from pathlib import Path 
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from PIL import Image
import clip

# Load models once
text_model = SentenceTransformer("all-MiniLM-L6-v2")
#text_model = SentenceTransformer("all-MiniLM-L12-v2")
#text_model = SentenceTransformer("all-mpnet-base-v2")
clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
#clip_model, preprocess = clip.load("ViT-B/32", device="cpu")


import numpy as np

import re

def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

# --- Constants ---
K = 3  # Number of top results to return

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI()

HEALTH_ENDPOINT_NAME = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ENDPOINT_NAME = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


# --- Pydantic Models ---
class ImageBytes(BaseModel):
    b64: str
    # The image encoded as a base64 string.


class Instance(BaseModel):
    image_bytes: Optional[ImageBytes] = None
    # Optional: An object containing the base64 encoded image.
    text_input: Optional[str] = None
    # Optional: A string of text input.
    # One of image_bytes or text_input must be provided.


class PredictionRequest(BaseModel):
    instances: List[Instance]
    # A list of instances to process. Each instance can contain either an image (as base64) or text.


class PredictionResponseItem(BaseModel):
    ranked_documents: List[str] = Field(default_factory=list)
    # A list of the top K most relevant document titles (strings) based on the input.
    # The order should reflect the ranking, with the most relevant first.
    # If the input instance could not be processed, this list should be empty.


class PredictionResponse(BaseModel):
    predictions: List[PredictionResponseItem]
    # A list of prediction results, where each item corresponds to an instance in the input request.


# --- Data Loading and Encoding (Applicant Implementation) ---


# TODO: Implement function to load image features
def load_image_features() -> Optional[Dict[str, Any]]:
    """
    Loads pre-computed image features. This could be embeddings or features extracted
    using various techniques.

    Returns:
        A dictionary mapping image identifiers to their feature representations,
        or None if loading fails.
    """
    if hasattr(load_image_features, "_cache"):
        return load_image_features._cache

    import pickle
    from pathlib import Path

    try:
        base_dir = Path(__file__).resolve().parent  # .../app
        candidate_paths = [
            base_dir / "image_features.pkl",
            base_dir.parent / "app" / "image_features.pkl",
            Path("image_features.pkl"),
            Path("app") / "image_features.pkl",
        ]

        feature_path: Optional[Path] = None
        for p in candidate_paths:
            if p.exists():
                feature_path = p
                break

        if feature_path is None:
            logger.warning("Image features file not found (image_features.pkl).")
            load_image_features._cache = None
            return None

        with open(feature_path, "rb") as f:
            raw = pickle.load(f)

        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected a dict from pickle, got {type(raw).__name__} instead."
            )

        validated: Dict[str, Any] = {}
        for img_id, feats in raw.items():
            # Normalize key type (some pickles may store non-str keys).
            img_key = str(img_id)

            arr: np.ndarray
            if isinstance(feats, np.ndarray):
                arr = feats.astype(np.float32, copy=False)
            elif isinstance(feats, (list, tuple)):
                arr = np.asarray(feats, dtype=np.float32)
                # If we ended up with dtype=object (e.g., list of vectors), stack explicitly.
                if arr.dtype == np.object_:
                    arr = np.stack([np.asarray(x, dtype=np.float32) for x in feats], axis=0)
            elif isinstance(feats, dict) and "embedding" in feats:
                arr = np.asarray(feats["embedding"], dtype=np.float32)
            else:
                arr = np.asarray(feats, dtype=np.float32)

            # Keep supported shapes:
            # - (dim,) for single embedding
            # - (num_variants, dim) for multi-variant embeddings
            if arr.ndim not in (1, 2):
                # Fall back to flatten for unexpected shapes.
                arr = arr.reshape(-1).astype(np.float32, copy=False)

            validated[img_key] = arr

        load_image_features._cache = validated
        logger.info(
            f"Loaded image features from {feature_path} with {len(validated)} entries."
        )

        # Log one sample shape (if available) for quick debugging.
        if validated:
            sample = next(iter(validated.values()))
            logger.info(
                f"Image feature sample shape: {getattr(sample, 'shape', None)}"
            )

        return load_image_features._cache
    except Exception as e:
        logger.exception(f"Failed to load image features: {e}")
        load_image_features._cache = None
        return None

def load_text_features() -> Optional[Dict[str, Any]]:
    """
    Loads pre-computed text features. This could be embeddings or features generated
    using different NLP methods.

    Returns:
        A dictionary mapping text identifiers to their feature representations,
        or None if loading fails.
    """
    import os
    import pickle

    if hasattr(load_text_features, "_cache"):
        return load_text_features._cache

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(base_dir, "text_features.pkl"),
            os.path.join("app", "text_features.pkl"),
            "text_features.pkl",
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    load_text_features._cache = pickle.load(f)
                    logger.info(
                        f"Loaded text features from {path} with {len(load_text_features._cache)} entries."
                    )
                    return load_text_features._cache

        logger.warning("Text features file not found.")
        load_text_features._cache = None
        return None
    except Exception as e:
        logger.exception(f"Failed to load text features: {e}")
        load_text_features._cache = None
        return None

def load_image_to_blogpost_mappings() -> Optional[Dict[str, str]]:
    """
    Loads the mapping between image identifiers and blog post titles.

    Returns:
        A dictionary mapping image identifiers to corresponding blog post titles (strings),
        or None if loading fails.
    """
    import os
    import json

    if hasattr(load_image_to_blogpost_mappings, "_cache"):
        return load_image_to_blogpost_mappings._cache

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(base_dir, "img_to_blog.json"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw_mapping = json.load(f)

                mapping = {}
                for img_name, blog_value in raw_mapping.items():
                    if isinstance(blog_value, str):
                        mapping[img_name] = blog_value.replace(".json", "").rstrip("-").strip()

                load_image_to_blogpost_mappings._cache = mapping
                logger.info(
                    f"Loaded image-to-blog mappings from {path} with {len(mapping)} entries."
                )
                return mapping

        logger.warning("img_to_blog.json not found.")
        load_image_to_blogpost_mappings._cache = None
        return None
    except Exception as e:
        logger.exception(f"Failed to load image-to-blog mappings: {e}")
        load_image_to_blogpost_mappings._cache = None
        return None
    

def encode_image(image: Image.Image) -> Optional[Any]:
    """
    Encodes a PIL Image into a feature representation. This is only needed if features
    are not pre-computed for all possible input images.

    Args:
        image: The PIL Image to encode.

    Returns:
        A representation of the image's features, or None if encoding fails.
    """
    try:
        image = image.convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = clip_model.encode_image(image_input)

        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].astype(np.float32)
    except Exception as e:
        logger.exception(f"Failed to encode image: {e}")
        return None
#def encode_image(image: Image.Image) -> Optional[Any]:
 #   """
  #  Encodes a PIL Image into a feature representation. This is only needed if features
   # are not pre-computed for all possible input images.

    #Args:
     #   image: The PIL Image to encode.

    #Returns:
     #   A representation of the image's features, or None if encoding fails.
    #"""
    #try:
     #   import numpy as np
      #  import torch
       # import clip

       # if not hasattr(encode_image, "_model"):
         #   device = "cpu"
          #  model, preprocess = clip.load("ViT-B/16", device=device)
           # encode_image._model = model
            #encode_image._preprocess = preprocess
            #encode_image._device = device

        #image = image.convert("RGB")
        #image_input = encode_image._preprocess(image).unsqueeze(0).to(
         #   encode_image._device
        #)

        #with torch.no_grad():
        #   features = encode_image._model.encode_image(image_input)

        #features = features / features.norm(dim=-1, keepdim=True)
        #return features.cpu().numpy()[0].astype(np.float32)
    #except Exception as e:
     #   logger.exception(f"Failed to encode image: {e}")
      #  return None
    
def encode_text(text: str) -> Optional[Any]:
    """
    Encodes a text string into a feature representation. This is only needed if features
    are not pre-computed for all possible input texts.

    Args:
        text: The text string to encode.

    Returns:
        A representation of the text's features, or None if encoding fails.
    """
    try:
        features = text_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return features.astype(np.float32)
    except Exception as e:
        logger.exception(f"Failed to encode text: {e}")
        return None
    
#def encode_text(text: str) -> Optional[Any]:
 #   """
  #  Encodes a text string into a feature representation. This is only needed if features
   # are not pre-computed for all possible input texts.

    #Args:
     #   text: The text string to encode.

    #Returns:
     #   A representation of the text's features, or None if encoding fails.
    #"""
    #try:
     #   features = text_model.encode(
      #      text,
       #     convert_to_numpy=True,
        #    normalize_embeddings=True,
        #)
        #return features.astype(np.float32)
    #except Exception as e:
     #   logger.exception(f"Failed to encode text: {e}")
      #  return None
#def encode_text(text: str) -> Optional[Any]:
 #   """
  #  Encodes a text string into a feature representation. This is only needed if features
   # are not pre-computed for all possible input texts.

    #Args:
     #   text: The text string to encode.

    #Returns:
     #   A representation of the text's features, or None if encoding fails.
    #"""
    #try:
     #   import numpy as np
      #  from sentence_transformers import SentenceTransformer

       # if not hasattr(encode_text, "_model"):
        #    encode_text._model = SentenceTransformer("all-mpnet-base-v2")

        #features = encode_text._model.encode(
         #   text,
          #  convert_to_numpy=True,
           # normalize_embeddings=True,
        #)
        #return features.astype(np.float32)
    #except Exception as e:
     #   logger.exception(f"Failed to encode text: {e}")
      #  return None


# --- Similarity and Ranking (Applicant Implementation) ---

# TODO: Implement function to calculate similarity between two feature representations
def calculate_similarity(feature1: Any, feature2: Any) -> float:
    """
    Calculates the similarity between two feature representations.

    Args:
        feature1: The first feature representation.
        feature2: The second feature representation.

    Returns:
        A float representing the similarity score.
    """
    try:
        f1 = np.asarray(feature1, dtype=np.float32)
        f2 = np.asarray(feature2, dtype=np.float32)

        if f2.ndim == 1:
            denom = np.linalg.norm(f1) * np.linalg.norm(f2)
            if denom == 0:
                return -1.0
            return float(np.dot(f1, f2) / denom)

        if f2.ndim == 2:
            f1_norm = np.linalg.norm(f1)
            f2_norms = np.linalg.norm(f2, axis=1)
            denoms = f1_norm * f2_norms

            valid = denoms > 0
            if not np.any(valid):
                return -1.0

            sims = np.full(f2.shape[0], -1.0, dtype=np.float32)
            sims[valid] = (f2[valid] @ f1) / denoms[valid]
            return float(np.max(sims))

        logger.warning(f"Unsupported feature shape: feature2.ndim={f2.ndim}")
        return -1.0

    except Exception as e:
        logger.exception(f"Failed to calculate similarity: {e}")
        return -1.0
#def calculate_similarity(feature1: Any, feature2: Any) -> float:
 #   """
  #  Calculates similarity between:
   # - query vector and single stored vector
    #- query vector and a stack of stored vectors (returns max similarity)
    #"""
    #try:
     #   f1 = np.asarray(feature1, dtype=np.float32)
      #  f2 = np.asarray(feature2, dtype=np.float32)

        ## Single vector case
       # if f2.ndim == 1:
        #    denom = np.linalg.norm(f1) * np.linalg.norm(f2)
         #   if denom == 0:
          #      return -1.0
         #   return float(np.dot(f1, f2) / denom)

        ## Multiple stored vectors case: return best match
        #if f2.ndim == 2:
         #   f1_norm = np.linalg.norm(f1)
          #  f2_norms = np.linalg.norm(f2, axis=1)
           # denoms = f1_norm * f2_norms

            #valid = denoms > 0
            #if not np.any(valid):
             #   return -1.0

            #sims = np.full(f2.shape[0], -1.0, dtype=np.float32)
            #sims[valid] = (f2[valid] @ f1) / denoms[valid]
            #return float(np.max(sims))

        #logger.warning(f"Unsupported feature shape: feature2.ndim={f2.ndim}")
        #return -1.0

    #except Exception as e:
     #   logger.exception(f"Failed to calculate similarity: {e}")
      #  return -1.0
#def calculate_similarity(feature1: Any, feature2: Any) -> float:
 #   """
  #  Calculates the similarity between two feature representations.

   # Args:
    #    feature1: The first feature representation.
     #   feature2: The second feature representation.

    #Returns:
     #   A float representing the similarity score.
    #"""
    #try:
     #   import numpy as np

      #  f1 = np.asarray(feature1, dtype=np.float32)
       # f2 = np.asarray(feature2, dtype=np.float32)

        #denom = np.linalg.norm(f1) * np.linalg.norm(f2)
        #if denom == 0:
         #   return -1.0

        #return float(np.dot(f1, f2) / denom)
    #except Exception as e:
     #   logger.exception(f"Failed to calculate similarity: {e}")
      #  return -1.0

# TODO: Implement function to get top K ranked items based on similarity
def get_top_k_ranked_items(
    similarities: Dict[str, float], k: int
) -> List[tuple[str, float]]:
    """
    Sorts items by their similarity scores and returns the top K items.

    Args:
        similarities: A dictionary mapping item identifiers to their similarity scores.
        k: The number of top items to return.

    Returns:
        A list of tuples, where each tuple contains the item identifier and its similarity score,
        sorted in descending order of similarity.
    """
    try:
        sorted_items = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        image_to_blog = load_image_to_blogpost_mappings()

        if image_to_blog is None:
            return sorted_items[:k]

        results = []
        seen_blogposts = set()

        for item_id, score in sorted_items:
            if item_id in image_to_blog:
                blog_slug = image_to_blog[item_id]
                if blog_slug in seen_blogposts:
                    continue
                seen_blogposts.add(blog_slug)
                results.append((item_id, score))
            else:
                if item_id in seen_blogposts:
                    continue
                seen_blogposts.add(item_id)
                results.append((item_id, score))

            if len(results) == k:
                break

        return results

    except Exception as e:
        logger.exception(f"Failed to rank items: {e}")
        return []



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get(HEALTH_ENDPOINT_NAME, status_code=200)
def health():
    return {"status": "OK"}


@app.post(PREDICT_ENDPOINT_NAME, response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> dict:
    """
    Processes a list of input instances to find the top K most relevant blog post titles.

    For each instance in the request:
    - If 'image_bytes' is provided, it should be decoded, its features obtained (either by
      loading pre-computed features or encoding the image), and compared against existing
      image features to find the top K most similar images. The corresponding blog post
      titles for these top images should be retrieved.
    - If 'text_input' is provided, its features should be obtained (either by loading
      pre-computed features or encoding the text), and compared against existing text features
      to find the top K most similar text entries. The identifiers of these top text entries
      should then be used to retrieve the corresponding blog post titles (you might need an
      additional mapping for this).
    - If neither 'image_bytes' nor 'text_input' is provided, the 'ranked_documents' for that
      instance should be an empty list.
    - Errors during processing should be handled gracefully, resulting in an empty
      'ranked_documents' list for the affected instance.

    Args:
        request (PredictionRequest): A Pydantic model containing a list of instances.
                                     Each instance has optional 'image_bytes' and 'text_input' fields.

    Returns:
        dict: A dictionary conforming to the PredictionResponse model, containing a list of
              PredictionResponseItem. Each item holds a list of 'ranked_documents' (strings).
    """
    predictions = []

    # --- Load necessary features and mappings ---
    image_features = load_image_features()
    text_features = load_text_features()
    image_to_blogpost_titles = load_image_to_blogpost_mappings()

    for i, instance in enumerate(request.instances):
        ranked_documents = []
        try:
            if instance.image_bytes and instance.image_bytes.b64:
                # --- Process Image Instance ---
                if image_features is None or image_to_blogpost_titles is None:
                    logger.warning(
                        f"Instance {i}: Skipping image prediction - features or mappings not loaded."
                    )
                    predictions.append(PredictionResponseItem())
                    continue

                try:
                    img_data = base64.b64decode(instance.image_bytes.b64)
                    query_image = Image.open(BytesIO(img_data))
                except (
                    base64.binascii.Error,
                    UnidentifiedImageError,
                    Exception,
                ) as decode_err:
                    logger.warning(
                        f"Instance {i}: Failed to decode base64 image: {decode_err}"
                    )
                    predictions.append(PredictionResponseItem())
                    continue

                # Get query image features
                query_image_features = (
                    encode_image(query_image)
                    if encode_image
                    else image_features.get("some_placeholder_key")
                )  # Example
                if query_image_features is None:
                    logger.warning(
                        f"Instance {i}: Failed to obtain features for query image."
                    )
                    predictions.append(PredictionResponseItem())
                    continue

                similarities = {}
                for img_identifier, stored_features in image_features.items():
                    similarity = calculate_similarity(
                        query_image_features, stored_features
                    )
                    similarities[img_identifier] = similarity

                sorted_images_with_scores = get_top_k_ranked_items(
                    similarities, K
                )

                for img_identifier, _ in sorted_images_with_scores:
                    title = image_to_blogpost_titles.get(img_identifier)
                    if title:
                        ranked_documents.append(title)
                    else:
                        logger.warning(
                            f"Instance {i}: No blog post title found for image: {img_identifier}"
                        )

            elif instance.text_input:
                # --- Process Text Instance ---
                if text_features is None:
                    logger.warning(
                        f"Instance {i}: Skipping text prediction - text features not loaded."
                    )
                    predictions.append(PredictionResponseItem())
                    continue

                # Get query text features
                query_text_features = (
                    encode_text(instance.text_input)
                    if encode_text
                    else text_features.get("some_placeholder_key")
                )  # Example
                if query_text_features is None:
                    logger.warning(
                        f"Instance {i}: Failed to obtain features for query text."
                    )
                    predictions.append(PredictionResponseItem())
                    continue

                similarities = {}
                for text_identifier, stored_features in text_features.items():
                    similarity = calculate_similarity(
                        query_text_features, stored_features
                    )
                    similarities[text_identifier] = similarity

                sorted_texts_with_scores = get_top_k_ranked_items(
                    similarities, K
                )

                # Retrieve blog post titles based on the top K text identifiers
                # This might require loading an additional mapping.
                for text_identifier, _ in sorted_texts_with_scores:
                    ranked_documents.append(text_identifier)

            else:
                logger.warning(
                    f"Instance {i}: Invalid instance format - missing image_bytes or text_input."
                )

            predictions.append(
                PredictionResponseItem(ranked_documents=ranked_documents)
            )

        except Exception as e:
            logger.exception(
                f"Instance {i}: Unexpected error during prediction: {e}"
            )
            predictions.append(PredictionResponseItem())

    if len(predictions) != len(request.instances):
        logger.error(
            f"Critical error: Mismatch between instance count ({len(request.instances)}) and prediction count ({len(predictions)})."
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction processing.",
        )

    return {"predictions": predictions}
