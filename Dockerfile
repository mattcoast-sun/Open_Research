# Production Dockerfile for IBM Code Engine
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for SQLite database
RUN mkdir -p /app/data

# Configure writable cache locations for Hugging Face/Transformers
# and SentenceTransformers. Keep caches under /app (owned by appuser).
ENV HF_HOME=/app/.cache/hf \
    TRANSFORMERS_CACHE=/app/.cache/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/app/.cache/hf/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

# Pre-create cache and logs directories
RUN mkdir -p /app/.cache/hf/transformers /app/.cache/hf/hub /app/.cache/sentence-transformers /app/logs

# Create the cloud ratings database during build
RUN python create_cloud_ratings_database.py

# Create non-root user for security (with home directory) and set HOME
RUN groupadd -r appuser && useradd -m -r -g appuser appuser
ENV HOME=/home/appuser
RUN chown -R appuser:appuser /app /home/appuser
USER appuser

# Expose port (IBM Code Engine standard)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Start the application with flexible port
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
