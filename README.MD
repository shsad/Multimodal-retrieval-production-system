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











## The Challenge

Your task is to implement a content retrieval engine as a FastAPI app that returns a list with the titles of the three ML6 blog posts that are most applicable for a query that's either text or image-based query. To achieve this, you'll have to:
1. Create feature vectors for the blogpost content (separate for the text and image content)
2. Encode the given text or image-based query
3. Find the three most relevant blogposts given a text or image-based query based on similarity


Your implementation will be tested using the `MRR@3` metric (more information [here](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)). Essentially, this metric will look for the position of the single, most relevant blogpost title corresponding to a query in your list of most similar blogposts. For any given query, you will receive a score of $\frac{1}{R}$, where R is the rank or position of the answer in your list of queries, if your list contains the answer. You only receive a score of 0 if the answer is not in your top 3. To succeed at this challenge, you'll have to achieve an `MRR@3` of 0.8 or higher.

### Implementing your design

We've provided a template for the FastAPI app in `app/main.py`. You'll need to modify this file to include your implementation of the retrieval engine. The `TODO` comments in the template file will guide you on what needs to be implemented. However, feel free to adjust the file in any way you see fit if the existing structure does not suit your implementation. 

ℹ️ A couple of tips during implementation:
- It's recommended to calculate and store features of the given blogpost texts and images in a separate script. This then allows you to simply load in these precomputed features in `app/main.py` during inference. If you do so, then make sure the features are stored in the `app` folder, as this folder will be added to the container for deployment later.
- If you decide to use pre-trained models, then make sure to load the models in an efficient way. I.e., load the models only once to then use them many times during evaluation.
- Although you have full implementation freedom, it might save you time if you stick to the existing skeleton structure and follow the TODOs.


### Testing your implementation locally

To test your implementation locally, you can use `app/evaluate.py`. This file runs through the test cases in `data/eval_data`, which you will have to generate yourself. We've already provided a few examples that align with our test cases that will be used for the final evaluation of your implementation, but it's your job to go through the ML6 blogposts and extend your local test set. The more test cases you add, the better your chances will be of having a representative performance indication.

⚠️ Note that the images we keep for evaluation seem to have been corrupted, which means your model can't just match pixels without creating features.

To run your local tests, you should first run your FastAPI app locally:
```bash
fastapi dev app/main.py
```

Then, you can execute the following command:

```bash
python app/evaluate.py
```

This command will evaluate your retrieval engine on both text and image queries. Again, you have complete freedom in how you implement the retrieval engine, as long as it meets the performance requirements. However, we recommend not to change `app/evaluate.py`, as this file aligns with the evaluation system we use to evaluate your implemenation on our own test cases. So, make sure that your implementation runs without errors on the local test cases in `app/evaluate.py`. 

ℹ️ A couple of quick tips for testing:
- We recommend to add at least 20-30 queries with a good balance between text queries and images to your local test set to get a representative performance indication. More is always better. An LLM might be handy to create more queries efficiently, but just make sure they're representative.
- When adding queries to your local test set, try to make them as specific as possible to clearly correspond to a single blog post.
- We do not expect you to be able to handle queries using metadata from the blogposts. In this challenge, we focus on just text from content blocks and images from the blogpost.
- The corrupted images used for final evaluation contain images with different rotations and greyscale versions of the blogpost images. Make sure your model can also handle greyscale images to avoid dimensionality errors.


## Deploying the Model

Once you've got the code working, you will need to deploy the model to Google Cloud to turn it into an API that can receive text or image queries and return the most relevant blogpost titles. To deploy your model, you'll first need to package it in a Docker container and then deploy the container to Vertex AI.

### Setting up Google Cloud

1. [Create a Google Cloud account](https://cloud.google.com/free) if you don't have one already. You'll need a credit card for verification, but you'll get [free credit from Google](https://cloud.google.com/free/docs/gcp-free-tier/#free-trial) to run your Vertex AI instance. 
   **Note:** If you are not eligible for the free trial, please reach out to us as you are responsible for the costs associated with the project.

2. [Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install) which includes the `gcloud` command-line tool.

3. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```
   This will open a browser window where you can sign in with your Google account.

3. [Create a new project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) or select an existing one:
   ```bash
   gcloud projects create <your-project-id>
   gcloud config set project <your-project-id>
   ```


4. [Enable the required APIs](https://cloud.google.com/endpoints/docs/openapi/enable-api):
   ```bash
   gcloud services enable artifactregistry.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```


5. [Create an Artifact Registry repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos) for your Docker images:
   ```bash
   gcloud artifacts repositories create blogpost-search \
   --repository-format=docker \
   --location=europe-west1 \
   --description="ML6 blogpost search docker repository"
   ```

### Build the Docker image

We've provided a Dockerfile that includes everything needed to package your FastAPI app. This Docker container will take care of installing all dependencies and setting up the environment to run your retrieval engine. Make sure that the `requirements.txt` file includes all the dependencies needed for your container.

To build and push your Docker image to Google Cloud:
```bash
gcloud builds submit --region=europe-west1 --tag=europe-west1-docker.pkg.dev/<your-project-id>/blogpost-search/retrieval-engine:latest
```

This can take a while to run.

### Deploy the Docker container to Vertex AI

Deploy your container to Vertex AI using the provided `deploy.py` script:
```bash
python deploy.py --gcp-project-id <your-project-id>
```

This can also take a while to run.

Alternatively, you can deploy manually through the [Google Cloud Console](https://console.cloud.google.com/):

1. Go to Vertex AI > Endpoints
2. Create a new endpoint in the `europe-west1` region
3. Deploy your container to the endpoint
We recommend using `europe-west1` as the regional endpoint if available. Note that the region of the bucket should correspond to the region of the model.

### Checking your Deployed Model

Before you submit your solution, you can check if your deployed model works correctly by executing the following commands:

```bash
ENDPOINT_ID=<your_endpoint_id>
PROJECT_ID=<your_project_id>

# Test with the evaluation data you created
gcloud ai endpoints predict $ENDPOINT_ID \
--project=$PROJECT_ID \
--region=europe-west1 \
--json-request=app/data/eval_data/deployment_test.json
```

Check if you are able to get a prediction out of the `gcloud` command. If you get errors, you should try to resolve them before submitting the solution.

The value for the `$ENDPOINT_ID` variable can be found in your project on the [Google Cloud Console](https://console.cloud.google.com/) under Vertex AI > Endpoints. You will need this value and your Google Cloud Project ID to submit your coding test.

## Submitting your Coding Test

Once you are able to execute the command above without errors, you can add us to your project:

* Go to the menu of your project
* Click *IAM & admin*
* Click *Add*
* Add `ml6-coding-challenge-evaluator@recruiting-220608.iam.gserviceaccount.com` as a member with the role *Project Owner*

After you added us to your project you should fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSdaG7LFxjBOxxD4aWJfrvU6cXv31UWy0zYfC1HJTK9B8J3jdA/viewform?usp=dialog) so we are able to automatically evaluate your solution to the coding test. Once you've filled in the form you should receive an email with the results within 2 hours. 

To be able to pass the coding test, you should be able to get an MRR@3 of 0.8 on our evaluation dataset. If your MRR@3 however seems to be less than 0.8 after we evaluated it, you can just keep submitting solutions until you are able to get an MRR@3 of 0.8. We hope with you that your results are good enough to land an interview at ML6.

If you are invited for an interview at ML6 afterwards, make sure to bring your laptop with a copy of the code you wrote, so you can explain your solution to us.

## Troubleshooting

If you encounter issues with your deployed solution or receive an error notification in the evaluation email, and you are sure your model runs without issues locally, then checking out the logs on GCP would be a good first step to diagnose and resolve the problem. Access detailed execution logs by navigating to Google Cloud Console → Vertex AI → Endpoints → Your endpoint → Your deployed model
and clicking on the link pointing to the relevant logs.

## Taking down the deployed model

After you have received the evaluation email, we no longer require access to the model. Please check the corresponding [documentation page](https://cloud.google.com/vertex-ai/docs/predictions/undeploy-model) for removing Vertex AI endpoints and models. This can be done via the UI or the command line.