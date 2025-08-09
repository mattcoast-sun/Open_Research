# Research Query Processing Tools for watsonx Orchestrate

## Project Overview

This project implements a FastAPI application designed to serve as intelligent tools within IBM watsonx Orchestrate's Agent Development Kit (ADK). The application provides four interconnected endpoints that work together to process and answer complex research queries through a multi-stage pipeline.

## Architecture & Design

The system follows a staged processing approach where each tool handles a specific aspect of query processing:

```
User Query → Clarification → SQL Generation → Vector Search → Cohesive Answer
```

### Integration with watsonx Orchestrate

This application is specifically designed for integration with [IBM watsonx Orchestrate ADK](https://developer.watson-orchestrate.ibm.com/tools/create_tool#creating-openapi-based-tools) as OpenAPI-based tools. Each endpoint can be imported as a separate tool that agents can use to process research queries.

## Core Components

### 1. Clarification Agent (`/clarification-agent`)

**Purpose**: Refines user queries through iterative questioning to improve precision and clarity.

**Features**:
- **State Management**: Maintains conversation sessions using session IDs
- **Question Limit**: Maximum of 3 clarification questions per session
- **Progressive Refinement**: Each question builds on previous answers
- **EOT (End of Turn) Detection**: Automatically transitions when clarification is complete

**Input**:
- `user_query`: Initial or follow-up query from user
- `session_id`: Optional session identifier for state tracking
- `previous_questions`: History of questions asked
- `previous_answers`: User responses to previous questions

**Output**:
- `question`: Next clarification question (if needed)
- `status`: Current processing status (`needs_clarification`, `complete`)
- `clarified_query`: Final refined query when complete
- `session_id`: Session tracking identifier

**State Considerations**:
- Sessions are maintained in-memory (production should use persistent storage)
- Each session tracks original query, questions asked, and user responses
- Automatic progression to completion after 3 questions or when clarity is achieved

### 2. SQL Query Tool (`/sql-query-generator`)

**Purpose**: Converts natural language queries into structured SQL based on database schema.

**Features**:
- **Schema-Aware Generation**: Considers database structure and relationships
- **Query Optimization**: Generates efficient SQL with appropriate indexes and joins
- **Explanation**: Provides human-readable explanation of query logic
- **Confidence Scoring**: Estimates relevance and accuracy of generated SQL

**Input**:
- `clarified_query`: Refined user question from clarification stage
- `schema_context`: Optional additional schema information

**Output**:
- `sql_query`: Generated SQL statement
- `explanation`: Detailed explanation of query logic
- `estimated_relevance`: Confidence score (0-1) for query appropriateness

**Implementation Notes**:
- Current implementation provides stub SQL generation
- Production version should integrate with schema analysis and NLP-to-SQL engines
- Query validation and security checks should be implemented

### 3. Vector DB Query Tool (`/vector-search`)

**Purpose**: Performs semantic similarity search on vector database using both clarified query and SQL results.

**Features**:
- **Hybrid Search**: Combines semantic similarity with SQL filtering
- **Configurable Results**: Adjustable maximum result count
- **Rich Metadata**: Returns similarity scores and document metadata
- **Performance Tracking**: Provides search execution metrics

**Input**:
- `clarified_query`: User's refined research question
- `sql_results`: Optional results from SQL query for filtering
- `max_results`: Maximum number of results to return (default: 10)

**Output**:
- `results`: Array of matching documents with similarity scores
- `total_found`: Count of matching documents
- `search_metadata`: Performance and execution information

**Integration Points**:
- Designed to work with Elasticsearch vector search capabilities
- Can incorporate SQL results for enhanced filtering and ranking
- Supports various similarity algorithms (cosine, euclidean, etc.)

### 4. Cohesive Answer Tool (`/cohesive-answer`)

**Purpose**: Synthesizes vector search findings into comprehensive, well-structured answers.

**Features**:
- **Content Synthesis**: Combines multiple sources into coherent responses
- **Source Attribution**: Provides proper citations and references
- **Quality Assessment**: Confidence scoring for answer reliability
- **Additional Insights**: Suggests related research directions

**Input**:
- `original_query`: User's initial research question
- `vector_results`: Results from vector database search
- `sql_context`: Optional additional context from SQL queries

**Output**:
- `answer`: Comprehensive synthesized response
- `sources`: Structured list of sources and citations
- `confidence_score`: Answer quality assessment (0-1)
- `additional_recommendations`: Suggested follow-up research areas

## Technical Specifications

### API Framework
- **FastAPI**: Modern, fast web framework with automatic OpenAPI generation
- **Pydantic**: Type validation and serialization
- **Uvicorn**: ASGI server for production deployment

### Key Dependencies
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
elasticsearch>=7.17.0,<8.0.0
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
```

### Data Models

All endpoints use strongly-typed Pydantic models for request/response validation:

- **Request Models**: `ClarificationRequest`, `SQLQueryRequest`, `VectorSearchRequest`, `CohesiveAnswerRequest`
- **Response Models**: `ClarificationResponse`, `SQLQueryResponse`, `VectorSearchResponse`, `CohesiveAnswerResponse`
- **Enums**: `QueryStatus` for standardized status tracking

## watsonx Orchestrate Integration

### OpenAPI Specification
The FastAPI application automatically generates OpenAPI specifications at `/docs` and `/redoc` endpoints. These can be directly imported into watsonx Orchestrate as OpenAPI-based tools.

### Tool Configuration
Each endpoint should be configured as a separate tool in watsonx Orchestrate:

1. **Clarification Tool**: For initial query refinement
2. **SQL Generator Tool**: For structured data queries
3. **Vector Search Tool**: For semantic document retrieval
4. **Answer Synthesis Tool**: For final response generation

### Agent Workflow
A typical agent workflow would:
1. Use Clarification Tool to refine user queries
2. Generate SQL queries using SQL Generator Tool
3. Perform vector search using Vector Search Tool
4. Synthesize final answers using Answer Synthesis Tool

## Deployment Instructions

### Quick Start

1. **Ensure you have a .env file** with your credentials (see example_env.txt)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the server**:
   ```bash
   python run_server.py
   ```
4. **Test the endpoints**:
   ```bash
   python test_endpoints.py
   ```
5. **Access the API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Local Development
```bash
# Alternative ways to run
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Docker (recommended)
# Create Dockerfile and deploy to container platform
```

### Environment Configuration
Create a `.env` file for environment-specific settings:
```
ELASTICSEARCH_URL=https://your-elasticsearch-instance
ELASTICSEARCH_USERNAME=your-username
ELASTICSEARCH_PASSWORD=your-password
DATABASE_URL=postgresql://user:pass@localhost/db
```

## Current Implementation Status

### ✅ Completed
- ✅ **FastAPI application structure** with full OpenAPI integration
- ✅ **All four endpoints fully implemented** with real functionality
- ✅ **Elasticsearch Integration**: Connected to actual vector database with your uploaded data
- ✅ **OpenAI Integration**: Using GPT-4o-mini for intelligent processing
- ✅ **Real Clarification Agent**: AI-powered question generation and session management
- ✅ **Real SQL/ES Query Generation**: AI converts natural language to Elasticsearch queries
- ✅ **Real Vector Search**: Semantic similarity search using sentence transformers
- ✅ **Real Answer Synthesis**: AI-powered comprehensive answer generation
- ✅ **Session management** for clarification agent with state tracking
- ✅ **Type validation** with Pydantic models
- ✅ **Error handling** with fallback mechanisms
- ✅ **Performance monitoring** with timing metrics

### 🚧 To Be Implemented (Optional Enhancements)
- **Authentication**: Add security for production deployment
- **Persistent State**: Replace in-memory sessions with database storage
- **Advanced Caching**: Query result caching for performance
- **Unit Testing**: Comprehensive test suite
- **Rate Limiting**: API usage controls

## Future Enhancements

### Advanced Features
- **Multi-language Support**: Query processing in multiple languages
- **Custom Schema Upload**: Dynamic schema registration
- **Query Caching**: Cache frequent queries for performance
- **Analytics Dashboard**: Usage metrics and performance monitoring

### Integration Possibilities
- **External APIs**: Integration with academic databases, research repositories
- **Custom Models**: Support for domain-specific language models
- **Collaborative Features**: Multi-user sessions and shared queries

## Security Considerations

- **Input Validation**: All inputs are validated using Pydantic models
- **SQL Injection Prevention**: Parameterized queries and input sanitization
- **Rate Limiting**: Should be implemented for production use
- **Authentication**: OAuth2 or API key authentication recommended
- **Data Privacy**: Ensure compliance with data protection regulations

## Monitoring & Observability

### Recommended Monitoring
- **Application Performance**: Response times, error rates
- **Database Performance**: Query execution times, resource usage
- **Vector Search Metrics**: Similarity score distributions, search latency
- **User Behavior**: Query patterns, clarification effectiveness

### Logging Strategy
- **Structured Logging**: JSON format for better parsing
- **Request Tracing**: Unique request IDs for debugging
- **Performance Metrics**: Detailed timing for each processing stage

## Contributing

When extending this application:
1. Maintain type safety with Pydantic models
2. Follow FastAPI best practices for endpoint design
3. Ensure OpenAPI compatibility for watsonx Orchestrate
4. Add appropriate error handling and logging
5. Update documentation for new features

## License

[Specify your license here]

---

*This project is designed to leverage the power of IBM watsonx Orchestrate's Agent Development Kit for creating intelligent, multi-stage research query processing capabilities.*
