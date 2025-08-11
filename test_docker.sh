#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "No .env file found!"
    exit 1
fi

# Stop and remove any existing container
docker stop test-torch-free 2>/dev/null || true
docker rm test-torch-free 2>/dev/null || true

echo "Starting torch-free container with environment variables..."

# Run Docker with all necessary environment variables
docker run -d \
    --name test-torch-free \
    -p 8080:8080 \
    -e EMBEDDINGS_MODE=openai \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e ES_URL="$ES_URL" \
    -e ES_USER="$ES_USER" \
    -e ES_PASS="$ES_PASS" \
    -e CLOUD_ID="$CLOUD_ID" \
    -e API_KEY="$API_KEY" \
    -e INDEX="$INDEX" \
    open-research:torch-free

echo "Container started. Checking status..."
sleep 3

# Check if container is running
if docker ps | grep test-torch-free > /dev/null; then
    echo "✅ Container is running successfully!"
    echo "🌐 Testing health endpoint..."
    sleep 2
    curl -s http://localhost:8080/health | jq . || echo "Health check response (raw):"
else
    echo "❌ Container failed to start. Checking logs..."
    docker logs test-torch-free
fi
