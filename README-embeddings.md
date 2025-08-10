# Embedding Configuration for Code Engine

This branch (`feature/openai-embeddings`) supports multiple embedding backends for query processing without requiring torch installation.

## Embedding Modes

### 1. HF (Hugging Face Inference API) - **RECOMMENDED for Code Engine**
```env
EMBEDDINGS_MODE=hf
HF_API_TOKEN=your_hf_token
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

**Benefits:**
- ✅ No torch/ML dependencies needed
- ✅ 384 dimensions (matches existing index)
- ✅ Same model as your documents
- ✅ Fast deployment on Code Engine

**Setup:**
1. Get HF token: https://huggingface.co/settings/tokens
2. Set `HF_API_TOKEN` in Code Engine secrets
3. Uses public Inference API (free tier available)

### 2. OpenAI Embeddings
```env
EMBEDDINGS_MODE=openai
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Note:** 1536 dims vs your 384-dim index → falls back to text search

### 3. Local (original)
```env
EMBEDDINGS_MODE=local
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

**Note:** Requires torch → not recommended for Code Engine

## Docker Build

```bash
# Torch-free build (default)
docker build -t myapp:torch-free .

# Or explicitly specify
docker build --build-arg REQUIREMENTS_FILE=requirements.openai.txt -t myapp:torch-free .

# Full ML build (if needed)
docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t myapp:full .
```

## Code Engine Deployment

1. Use `requirements.openai.txt` (default in Dockerfile)
2. Set environment variables:
   ```
   EMBEDDINGS_MODE=hf
   HF_API_TOKEN=<your-token>
   OPENAI_API_KEY=<your-openai-key>
   ```
3. Much smaller image, faster cold starts

## Validation

The system automatically:
- Checks embedding dimensions against Elasticsearch mapping
- Falls back to text search if dimensions don't match
- Logs warnings for dimension mismatches
