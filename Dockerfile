# Force Debian 12 (Bookworm) to avoid 2025/2026 Debian 13 package issues
FROM python:3.11.14-slim-bookworm

# Set working directory
WORKDIR /code

# Standard setup for Hugging Face Spaces
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy specific files with correct permissions
COPY --chown=user helper.py $HOME/app/helper.py
COPY --chown=user cleaned_embeddings_dataframe.pkl $HOME/app/cleaned_embeddings_dataframe.pkl
COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
