# Multimodal Retrieval Production System

Clone repository: 

```bash
git clone https://github.com/shsad/Multimodal-retrieval-production-system
```

## Overview 

This repository implements a **production-ready multimodal content retrieval engine** for blog posts.

The system accepts **text or image queries** and returns the **top 3 most relevant blog post titles**, based on semantic similarity. It combines modern **NLP and computer vision techniques** to enable unified retrieval across modalities.

The project is designed with **scalability and deployment in mind**, featuring:

 - Precomputed embeddings for efficient inference

 - A FastAPI-based serving layer

 - Containerization via Docker

 - Deployment on Google Cloud Vertex AI

## Problem Setting

Given a collection of blog posts (including text and images), the system:

 1. Encodes blog post content into feature vectors
 2. Encodes incoming queries (text or image)
 3. Computes similarity in a shared embedding space
 4. Returns the **top 3 most relevant blog posts**

The system is evaluated using the **MRR@3 (Mean Reciprocal Rank)** metric.

## Data

The project follows a structured data format:

data/
│── orig_data/
│   ├── blogposts/        # JSON blogpost content
│   ├── images/           # Blogpost images
│   ├── img_to_blog.json  # Image-to-blog mapping
│
│── eval_data/
│   ├── evaluation_data.json
│   ├── eval_images/
│   ├── deployment_test.json

## Data Source

The system is data-agnostic and can be used with:

- Personal blog datasets
- Public blog repositories
- Scraped website content

To use your own data:

- Provide blog posts in JSON format (title + content blocks)
- Store associated images
- Create an image-to-blog mapping (`img_to_blog.json`)

## System Architecture

The retrieval pipeline consists of:

### 1. Feature Engineering (Offline)

- Text embeddings generated using transformer-based models  
- Image embeddings generated using vision models  
- Features are precomputed and stored for fast retrieval  

### 2. Query Encoding (Online)

- Text queries → encoded via text encoder  
- Image queries → encoded via image encoder  

### 3. Retrieval

- Similarity search (e.g., cosine similarity)  
- Ranking of blog posts  
- Top-3 results returned  

### 4. API Layer

- Implemented using FastAPI  
- Handles both text and image queries  

## Implementation Details

The system is implemented as a production-ready pipeline, extending the original skeleton into a complete solution:

- Precomputed embeddings stored in the app/ directory
- Efficient model loading (loaded once at startup)
- Unified embedding space for text and images
- Robust handling of image variations (e.g., grayscale, rotation)
- Modular design separating preprocessing, inference, and serving

The main application is located in:

```bash
app/main.py
```

## Local Testing

## Start the API
```bash
fastapi dev app/main.py
```

## Run Evaluation
```bash
python app/evaluate.py
```

## Evaluation Setup
 - The evaluation uses queries defined in 
  ```data/eval_data/evaluation_data.json````
 - Includes both:
   - Text queries
   - Image queries

## Notes
 - The evaluation dataset contains **custom-crafted queries**
 - Queries are designed to map clearly to specific blog posts
 - The system is tested against **corrupted/modified images** (e.g., grayscale, rotated), ensuring robustness

# Deployment

The system is fully containerized and deployable on **Google Cloud Vertex AI**

 1. ## Setup Google Cloud

 ```bash
 gcloud auth login
 gcloud config set project <your-project-id>

 gcloud services enable artifactregistry.googleapis.com
 gcloud services enable cloudbuild.googleapis.com
 gcloud services enable aiplatform.googleapis.com
 ```

2. ## Create Artifact Registry
 ```bash
 gcloud artifacts repositories create blogpost-search \
 --repository-format=docker \
 --location=europe-west1 \
 --description="Blogpost retrieval system"
 ```

3. ## Build & Push Docker Image
 ```bash
 gcloud builds submit --region=europe-west1 \
 --tag=europe-west1-docker.pkg.dev/<your-project-id>/blogpost-search/retrieval-engine:latest
 ```

4. ## Deploy to Vertex AI
 ```bash
 python deploy.py --gcp-project-id <your-project-id>
 ```

5. ## Test Deployed Endpoint
 ```bash 
 ENDPOINT_ID=<your_endpoint_id>
 PROJECT_ID=<your_project_id>

 gcloud ai endpoints predict $ENDPOINT_ID \
 --project=$PROJECT_ID \
 --region=europe-west1 \
 --json-request=app/data/eval_data/deployment_test.json
 ``` 

# Key Features
 - Multimodal retrieval (text + image)
 - Fast inference via precomputed embeddings
 - Transformer-based representations
 - Dockerized for reproducibility
 - Deployable on Vertex AI
 - Evaluation using MRR@3

# Possible Extensions

 - Use **CLIP-style joint embedding models** for improved cross-modal alignment

 - Add ** vector databases** (FAISS, Pinecone, Weaviate) for scalability

 - Improve ranking via **re-ranking models (cross-encoders)**

 - Add **query expansion using LLMs**

 - Extend to **full RAG pipeline for blog Q&A**











