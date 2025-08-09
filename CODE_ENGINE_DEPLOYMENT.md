# IBM Code Engine Deployment Guide

## 🚀 Quick Deploy to IBM Code Engine

### Prerequisites
- IBM Cloud CLI installed and configured
- Code Engine CLI plugin installed
- Docker (for local testing)

### Step 1: Prepare Your Environment

1. **Clone/Download your repository**
2. **Set up your environment variables** (see `env.production.template`)
3. **Test locally with Docker**:
   ```bash
   docker build -t research-api .
   docker run -p 8000:8000 --env-file .env research-api
   ```

### Step 2: Deploy to Code Engine

#### Option A: Direct Source Deployment
```bash
# Login to IBM Cloud
ibmcloud login

# Target your Code Engine project
ibmcloud ce project select --name YOUR_PROJECT_NAME

# Deploy from source
ibmcloud ce application create \
  --name research-query-api \
  --build-source . \
  --build-strategy dockerfile \
  --port 8000 \
  --env-from-secret elasticsearch-creds \
  --env-from-secret openai-creds \
  --env PORT=8000 \
  --env DB_PATH=/app/data/cloud_ratings.db \
  --env LOG_LEVEL=INFO \
  --cpu 1 \
  --memory 2G \
  --min-scale 0 \
  --max-scale 10
```

#### Option B: Container Registry Deployment
```bash
# Build and push to IBM Container Registry
ibmcloud cr build -t us.icr.io/YOUR_NAMESPACE/research-api .

# Deploy from registry
ibmcloud ce application create \
  --name research-query-api \
  --image us.icr.io/YOUR_NAMESPACE/research-api \
  --port 8000 \
  --env-from-secret elasticsearch-creds \
  --env-from-secret openai-creds \
  --env PORT=8000 \
  --env DB_PATH=/app/data/cloud_ratings.db \
  --cpu 1 \
  --memory 2G \
  --min-scale 0 \
  --max-scale 10
```

### Step 3: Create Secrets

#### Elasticsearch Credentials
```bash
ibmcloud ce secret create \
  --name elasticsearch-creds \
  --from-literal ES_URL=https://your-elasticsearch-url \
  --from-literal ES_USER=your-username \
  --from-literal ES_PASS=your-password \
  --from-literal ES_PORT=31159
```

#### OpenAI Credentials
```bash
ibmcloud ce secret create \
  --name openai-creds \
  --from-literal OPENAI_API_KEY=your-openai-api-key
```

### Step 4: Verify Deployment

1. **Get application URL**:
   ```bash
   ibmcloud ce application get --name research-query-api
   ```

2. **Test health endpoint**:
   ```bash
   curl https://your-app-url/health
   ```

3. **Test API documentation**:
   ```
   https://your-app-url/docs
   ```

### Step 5: Test Enhanced Research Pipeline

```bash
curl -X POST "https://your-app-url/research-pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the best cloud providers for AI workloads?",
    "skip_clarification": true,
    "max_results": 5
  }'
```

## 📊 Application Features

### Available Endpoints
- `/health` - Health check
- `/docs` - API documentation  
- `/clarification-agent` - Query refinement
- `/sql-query-generator` - Elasticsearch query generation
- `/cloud-ratings-sql` - Cloud ratings SQL generation
- `/cloud-ratings-sql-execute` - SQL execution with results
- `/vector-search` - Semantic similarity search
- `/cohesive-answer` - Comprehensive answer synthesis
- `/research-pipeline` - Complete research workflow

### Key Capabilities
- ✅ **Enhanced Research Pipeline**: Combines vector search + SQL data
- ✅ **Cloud Ratings Analysis**: 11 cloud providers with detailed metrics
- ✅ **Vector Search**: Semantic search across cloud knowledge base
- ✅ **AI-Powered**: OpenAI integration for intelligent responses
- ✅ **Production Ready**: Logging, health checks, error handling

## 🔧 Configuration Options

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `DB_PATH` | SQLite database path | /app/data/cloud_ratings.db |
| `ES_URL` | Elasticsearch URL | Required |
| `ES_USER` | Elasticsearch username | Required |
| `ES_PASS` | Elasticsearch password | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LOG_LEVEL` | Logging level | INFO |

### Scaling Configuration
- **CPU**: 0.25 - 2 vCPU
- **Memory**: 512M - 4G
- **Min Scale**: 0 (scales to zero)
- **Max Scale**: 10 instances
- **Request Timeout**: 300s

## 🔍 Monitoring & Troubleshooting

### Health Checks
- **Endpoint**: `/health`
- **Expected Response**: `{"status": "healthy", "timestamp": "..."}`

### Logs
```bash
# View application logs
ibmcloud ce application logs --name research-query-api --follow
```

### Common Issues

1. **Elasticsearch Connection Failed**
   - Check ES_URL, ES_USER, ES_PASS environment variables
   - Verify network connectivity
   - Check Elasticsearch server status

2. **Database Not Found**
   - Database is created during container build
   - Check DB_PATH environment variable
   - Verify container has write permissions

3. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is set correctly
   - Check API usage limits
   - Monitor rate limiting

### Performance Optimization
- Use min-scale=1 for production to avoid cold starts
- Increase memory for large vector operations
- Monitor CPU usage for AI model inference

## 🎯 Next Steps

1. **Set up monitoring** with IBM Cloud Monitoring
2. **Configure custom domains** 
3. **Set up CI/CD pipelines**
4. **Add load testing**
5. **Implement caching** for frequent queries

## 📞 Support

For issues with:
- **Application**: Check logs and health endpoints
- **Code Engine**: IBM Cloud support
- **Elasticsearch**: Verify connection and credentials
- **OpenAI**: Check API status and limits
