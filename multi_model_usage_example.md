# Multi-Model Embedding Usage Guide

This guide shows how to use the enhanced upload script to maintain multiple embedding models in the same Elasticsearch indices.

## Basic Usage Examples

### 1. Add OpenAI Embeddings to Existing Data

Keep your existing sentence-transformers embeddings and add OpenAI embeddings:

```bash
# Set environment variables
export ACTIVE_MODELS="v2"  # Only add OpenAI embeddings
export UPDATE_EXISTING="true"  # Update existing documents
export RECREATE_INDICES="false"  # Don't delete existing data
export OPENAI_API_KEY="your_openai_key"

# Run the upload
python upload_data_multi_model.py
```

This will:
- ✅ Keep all existing `body_vector` fields (384-dim sentence-transformers)
- ✅ Add new `body_vector_v2` fields (1536-dim OpenAI) 
- ✅ Preserve all existing document data

### 2. Add Multiple Models Simultaneously

```bash
export ACTIVE_MODELS="v1,v2,v3"  # All three models
export UPDATE_EXISTING="true"
export RECREATE_INDICES="false"
export OPENAI_API_KEY="your_openai_key"

python upload_data_multi_model.py
```

### 3. Fresh Install with Multiple Models

```bash
export ACTIVE_MODELS="v1,v2"
export UPDATE_EXISTING="false"
export RECREATE_INDICES="true"  # Start fresh
export OPENAI_API_KEY="your_openai_key"

python upload_data_multi_model.py
```

## Document Structure After Multi-Model Upload

Your documents will have multiple vector fields:

```json
{
  "provider_name": "Amazon Web Services (AWS)",
  "overview": "AWS is the world's largest cloud platform...",
  "advantages": [...],
  "body_vector": [0.1, 0.2, ...],        // 384-dim sentence-transformers
  "body_vector_v2": [0.05, 0.15, ...],   // 1536-dim OpenAI
  "body_vector_v3": [0.08, 0.12, ...]    // 768-dim mpnet (if enabled)
}
```

## Model Configuration

The script supports these pre-configured models:

| Model Key | Type | Model Name | Dimensions | Vector Field |
|-----------|------|------------|------------|--------------|
| `v1` | sentence-transformers | all-MiniLM-L6-v2 | 384 | `body_vector` |
| `v2` | openai | text-embedding-3-small | 1536 | `body_vector_v2` |
| `v3` | sentence-transformers | all-mpnet-base-v2 | 768 | `body_vector_v3` |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ACTIVE_MODELS` | Comma-separated model keys to process | `"v1"` |
| `UPDATE_EXISTING` | Update existing documents vs create new | `"true"` |
| `RECREATE_INDICES` | Delete and recreate indices (⚠️ dangerous) | `"false"` |
| `OPENAI_API_KEY` | Required for OpenAI models | - |

## Safety Features

- **Preserves existing data**: When `UPDATE_EXISTING=true`, existing vector fields are preserved
- **Graceful failures**: If one model fails, others continue processing
- **Detailed logging**: Shows which embeddings were generated successfully
- **Index validation**: Updates mappings to support new vector fields

## Migration Strategies

### Strategy A: Gradual Migration
1. Add new embeddings alongside existing ones
2. Update search logic to use both models
3. A/B test performance between models
4. Eventually remove old embeddings

### Strategy B: Side-by-Side Comparison
1. Keep multiple models permanently
2. Allow users to choose embedding model
3. Compare search quality across models
4. Use ensemble methods for best results

## Performance Considerations

- **Storage**: Each additional model increases storage by ~vector_size × num_documents
- **Indexing time**: Multiple models mean multiple API calls during upload
- **Memory**: sentence-transformers models load into memory
- **API costs**: OpenAI embeddings have per-token pricing

## Troubleshooting

### Missing Dependencies
```bash
pip install openai sentence-transformers
```

### OpenAI API Issues
- Verify `OPENAI_API_KEY` is set correctly
- Check API quota and billing
- Monitor rate limits

### Elasticsearch Mapping Conflicts
- If mapping conflicts occur, consider `RECREATE_INDICES=true` (⚠️ loses data)
- Or manually update mappings via Elasticsearch API

### Memory Issues with Large Models
- Use OpenAI API instead of local models for memory-constrained environments
- Process in smaller batches
- Consider using sentence-transformers with `device='cpu'`
