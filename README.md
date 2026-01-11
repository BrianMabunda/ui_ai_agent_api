# 🤖  UI Complaints Agent API
A containerized AI service built with FastAPI and Python 3.11. This agent processes user complaints by querying a vector database of pre-calculated embeddings.

# 📁 Repository Structure

*    app.py - The FastAPI web server.
*    helper.py - Core logic for search and processing.
*    cleaned_embeddings_dataframe.pkl - The vector database (stored via Git LFS).
*    Dockerfile - Production-ready environment config.
*    requirements.txt - Python dependencies.


# 🚀 How to Run Locally
Follow these steps to get the API running on your machine using Docker.
## 1. Prerequisites
*    Docker Desktop installed and running.
*    Git LFS installed (required to pull the .pkl database file).
## 2. Clone the Repository

bash
```
git clone github.com
cd ui_ai_agent
git lfs pull
```
## 3. Setup Environment Variables
Create a file named .env in the root folder and add your Hugging Face token:
env

bash
```
HF_TOKEN=hf_your_token_here
```

## 4. Run with Docker (The Fast Way)
The easiest way to run the server is using Docker Compose:

bash
```
docker compose up --build

```


The API will now be live at: http://localhost:7860

# 🛠️ Manual Docker Commands
If you don't want to use Compose, you can use standard Docker commands:
Build the image:

bash
```
docker build -t ui-agent-api .

```

Run the container:

bash
```
docker run -p 7860:7860 --env-file .env ui-agent-api
```


# 🔌 API Endpoints
Once the server is running, you can access:
* Interactive Docs (Swagger): http://localhost:7860/docs
* Health Check: GET http://localhost:7860/
# ⚠️ Important for Contributors

*  Large Files: The .pkl file is tracked via Git LFS. If the file appears as a small text file instead of a data file, ensure you have run git lfs pull.
*  Python Version: This project is locked to Python 3.11.14 for consistency with the embedding model dependencies.

