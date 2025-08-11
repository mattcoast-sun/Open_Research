"""
FastAPI application for watsonx Orchestrate Agent Tools
=====================================================

This application provides four key endpoints designed to work as tools
within IBM watsonx Orchestrate Agent Development Kit (ADK):

1. Clarification Agent - Refines user queries through iterative questioning
2. SQL Query Tool - Converts natural language to SQL queries
3. Vector DB Query Tool - Performs similarity search on vector database
4. Cohesive Answer Tool - Generates final comprehensive responses

Each endpoint is designed to be imported as an OpenAPI-based tool in watsonx Orchestrate.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
from datetime import datetime
import os
import json
import sqlite3
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import openai
from openai import OpenAI
from system_prompt import system_prompt
from embeddings import EMBEDDINGS_MODE, get_query_embedding, get_dense_vector_dims

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/app.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Research Query Processing Tools",
    description="AI-powered tools for processing and answering research queries through watsonx Orchestrate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# Configuration and Initialization
# ============================================================================

def get_first_env(names: List[str], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default

# Elasticsearch configuration
ES_URL = get_first_env([
    "ES_URL", "ELASTICSEARCH_URL", "ELASTIC_URL", "ELASTIC_HOST",
    "ELASTICSEARCH_HOST", "ELASTICSEARCH_HOSTS"
], "http://localhost:9200")
ES_USER = get_first_env([
    "ES_USER", "ELASTICSEARCH_USERNAME", "ELASTIC_USER", "ELASTIC_USERNAME"
]) or "elastic"
ES_PASS = get_first_env([
    "ES_PASS", "ELASTICSEARCH_PASSWORD", "ELASTIC_PASSWORD", "ELASTIC_PASS"
]) or "your_password"
ES_PORT = get_first_env(["ES_PORT", "ELASTICSEARCH_PORT", "ELASTIC_PORT"], "").strip()

# Cloud configuration
CLOUD_ID = get_first_env(["CLOUD_ID", "ELASTIC_CLOUD_ID"]) or None
API_KEY = get_first_env(["ES_API_KEY", "ELASTIC_API_KEY", "API_KEY"]) or None

# Index configuration
BASE_INDEX = os.getenv("INDEX", "cloud_providers")
CLOUDS_INDEX = f"{BASE_INDEX}_providers" if BASE_INDEX != "providers" else "cloud_providers"
BEST_PRACTICES_INDEX = f"{BASE_INDEX}_best_practices" if BASE_INDEX != "providers" else "cloud_best_practices"

# Model configuration (for reference only when using local ST)
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Elasticsearch client with version compatibility
if ES_URL:
    ES_URL = ES_URL.rstrip("/")

try:
    if CLOUD_ID and API_KEY:
        # Try with version 8 compatibility first
        es = Elasticsearch(
            cloud_id=CLOUD_ID, 
            api_key=API_KEY,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
    else:
        if ES_PORT and ":" not in ES_URL.split("//", 1)[-1].split("/", 1)[0]:
            ES_URL = f"{ES_URL}:{ES_PORT}"
        es = Elasticsearch(
            ES_URL,
            http_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
    
    # Test the connection
    try:
        es_info = es.info()
        es_version = es_info.get('version', {}).get('number', 'unknown')
        logger.info(f"Connected to Elasticsearch {es_version}")
        print(f"✅ Connected to Elasticsearch {es_version}")
    except Exception as e:
        logger.warning(f"Elasticsearch connection test failed: {e}")
        print(f"⚠️ Elasticsearch connection test failed: {e}")
        
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {e}")
    print(f"❌ Failed to initialize Elasticsearch client: {e}")
    # Create a dummy client for graceful degradation
    es = None

# No local model initialization here; query embeddings are provided by embeddings module

# ============================================================================
# Data Models
# ============================================================================

class QueryStatus(str, Enum):
    """Status of query processing"""
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    NEEDS_CLARIFICATION = "needs_clarification"

class ClarificationRequest(BaseModel):
    """Request model for clarification agent"""
    user_query: str = Field(..., description="The user's initial or follow-up query")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation state")
    previous_questions: Optional[List[str]] = Field(default=[], description="Previously asked clarification questions")
    previous_answers: Optional[List[str]] = Field(default=[], description="User's previous answers")

class ClarificationResponse(BaseModel):
    """Response model for clarification agent"""
    session_id: str = Field(..., description="Session ID for tracking conversation")
    question: Optional[str] = Field(None, description="Next clarification question to ask user")
    status: QueryStatus = Field(..., description="Current status of clarification process")
    clarified_query: Optional[str] = Field(None, description="Final clarified query when complete")
    question_count: int = Field(..., description="Number of questions asked so far")

class SQLQueryRequest(BaseModel):
    """Request model for SQL query generation"""
    clarified_query: str = Field(..., description="The user's clarified research question")
    schema_context: Optional[str] = Field(None, description="Additional schema context if available")

class SQLQueryResponse(BaseModel):
    """Response model for SQL query generation"""
    sql_query: str = Field(..., description="Generated SQL query")
    explanation: str = Field(..., description="Explanation of the SQL query logic")
    estimated_relevance: float = Field(..., description="Confidence score for query relevance (0-1)")

class VectorSearchRequest(BaseModel):
    """Request model for vector database search"""
    clarified_query: str = Field(..., description="The user's clarified research question")
    sql_results: Optional[List[Dict[str, Any]]] = Field(default=[], description="Results from SQL query execution")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results to return")

class VectorSearchResponse(BaseModel):
    """Response model for vector database search"""
    results: List[Dict[str, Any]] = Field(..., description="Vector search results with similarity scores")
    total_found: int = Field(..., description="Total number of matching documents")
    search_metadata: Dict[str, Any] = Field(..., description="Metadata about the search execution")

class CohesiveAnswerRequest(BaseModel):
    """Request model for cohesive answer generation"""
    original_query: str = Field(..., description="The user's original research question")
    vector_results: List[Dict[str, Any]] = Field(..., description="Results from vector database search")
    sql_context: Optional[List[Dict[str, Any]]] = Field(default=[], description="Additional SQL query results")

class CohesiveAnswerResponse(BaseModel):
    """Response model for cohesive answer generation"""
    answer: str = Field(..., description="Comprehensive answer to the user's question")
    sources: List[Dict[str, str]] = Field(..., description="Sources and references used")
    confidence_score: float = Field(..., description="Confidence in the answer quality (0-1)")
    additional_recommendations: Optional[List[str]] = Field(default=[], description="Additional research suggestions")

class CloudRatingsSQLRequest(BaseModel):
    """Request model for cloud ratings SQL query generation"""
    question: str = Field(..., description="Natural language question about cloud ratings data")
    query_type: Optional[str] = Field(default="analysis", description="Type of query: analysis, comparison, ranking, or specific")

class CloudRatingsSQLResponse(BaseModel):
    """Response model for cloud ratings SQL query generation"""
    sql_query: str = Field(..., description="Generated SQLite query for cloud ratings database")
    explanation: str = Field(..., description="Explanation of what the query does")
    example_usage: str = Field(..., description="Example of how to execute the query")
    confidence_score: float = Field(..., description="Confidence in query accuracy (0-1)")
    query_type: str = Field(..., description="Detected type of query")
    estimated_results: str = Field(..., description="Description of expected results")

class CloudRatingsSQLExecutionRequest(BaseModel):
    """Request model for cloud ratings SQL query execution"""
    question: str = Field(..., description="Natural language question about cloud ratings data")
    execute_query: Optional[bool] = Field(default=True, description="Whether to execute the generated query")
    query_type: Optional[str] = Field(default="analysis", description="Type of query: analysis, comparison, ranking, or specific")

class CloudRatingsSQLExecutionResponse(BaseModel):
    """Response model for cloud ratings SQL query execution with results"""
    sql_query: str = Field(..., description="Generated SQLite query")
    query_results: List[Dict[str, Any]] = Field(..., description="Actual query execution results")
    column_names: List[str] = Field(..., description="Names of columns in the results")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time: float = Field(..., description="Query execution time in seconds")
    explanation: str = Field(..., description="Explanation of what the query does")
    confidence_score: float = Field(..., description="Confidence in query accuracy (0-1)")
    query_type: str = Field(..., description="Detected type of query")

# ============================================================================
# In-memory session storage (replace with persistent storage in production)
# ============================================================================
clarification_sessions: Dict[str, Dict] = {}

# ============================================================================
# Endpoint 1: Clarification Agent
# ============================================================================

@app.post("/clarification-agent", 
          response_model=ClarificationResponse,
          summary="Clarification Agent",
          description="Refines user queries through iterative questioning to improve precision")
async def clarification_agent(request: ClarificationRequest) -> ClarificationResponse:
    """
    The Clarification Agent asks targeted questions to refine and improve
    the precision of user queries. It can ask up to 3 questions before
    considering the query sufficiently clarified.
    """
    
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Retrieve or initialize session data
    if session_id not in clarification_sessions:
        clarification_sessions[session_id] = {
            "original_query": request.user_query,
            "questions_asked": [],
            "user_answers": [],
            "question_count": 0,
            "created_at": datetime.now()
        }
    
    session = clarification_sessions[session_id]
    
    # Update session with new information
    if request.previous_questions:
        session["questions_asked"].extend(request.previous_questions)
    if request.previous_answers:
        session["user_answers"].extend(request.previous_answers)
    
    # Check if we've reached the maximum number of questions
    if session["question_count"] >= 3:
        # Generate final clarified query using OpenAI
        try:
            conversation_history = ""
            for i, (q, a) in enumerate(zip(session["questions_asked"], session["user_answers"])):
                conversation_history += f"Q{str(i+1)}: {str(q)}\nA{str(i+1)}: {str(a)}\n"
            
            clarification_prompt = f"""
            {str(system_prompt)}
            
            You are the Clarification Agent. Based on the original query and the conversation history, 
            generate a final clarified query that incorporates all the insights gathered.
            
            Original Query: {str(session['original_query'])}
            
            Conversation History:
            {str(conversation_history)}
            
            Generate a comprehensive, clarified query that captures the user's intent with all the additional context:
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research clarification expert. Generate clear, comprehensive queries."},
                    {"role": "user", "content": clarification_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            clarified_query = response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to basic clarification
            clarified_query = f"Clarified query based on: {str(session['original_query'])} with additional context from {str(len(session['user_answers']))} clarification responses."
        
        return ClarificationResponse(
            session_id=session_id,
            question=None,
            status=QueryStatus.COMPLETE,
            clarified_query=clarified_query,
            question_count=session["question_count"]
        )
    
    # Generate next clarification question using OpenAI
    try:
        conversation_history = ""
        for i, (q, a) in enumerate(zip(session["questions_asked"], session["user_answers"])):
            conversation_history += f"Q{str(i+1)}: {str(q)}\nA{str(i+1)}: {str(a)}\n"
        
        question_prompt = f"""
        {str(system_prompt)}
        
        You are the Clarification Agent (Step 1). Your goal is to ask targeted questions to clarify the user's research needs.
        
        Original Query: {str(session['original_query'])}
        Questions asked so far: {str(session['question_count'])}/3
        
        Conversation History:
        {str(conversation_history)}
        
        Generate the next clarification question. Focus on:
        - Technical details and constraints
        - Specific use cases or requirements  
        - Time periods, scope, or scale
        - Implementation preferences
        
        Ask ONE specific, actionable question:
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a research clarification expert. Ask focused, specific questions to understand user needs."},
                {"role": "user", "content": question_prompt}
            ],
            max_tokens=150,
            temperature=0.4
        )
        
        next_question = response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback questions
        fallback_questions = [
            "Could you specify the time period or timeline you're interested in for this research?",
            "What specific aspects, features, or dimensions of this topic are most important to your use case?",
            "Are you looking for implementation guidance, comparative analysis, or strategic recommendations?"
        ]
        next_question = fallback_questions[session["question_count"]]
    
    session["question_count"] += 1
    session["questions_asked"].append(next_question)
    
    return ClarificationResponse(
        session_id=session_id,
        question=next_question,
        status=QueryStatus.NEEDS_CLARIFICATION,
        clarified_query=None,
        question_count=session["question_count"]
    )

# ============================================================================
# Endpoint 2: SQL Query Tool
# ============================================================================

@app.post("/sql-query-generator",
          response_model=SQLQueryResponse,
          summary="SQL Query Generator",
          description="Converts natural language queries into SQL based on appropriate schema")
async def sql_query_generator(request: SQLQueryRequest) -> SQLQueryResponse:
    """
    The SQL Query Tool analyzes a user's clarified question and converts it
    into an appropriate SQL query based on the available database schema.
    """
    
    # Define the schema context for our cloud data
    schema_context = f"""
    Available Elasticsearch indices:
    
    1. {CLOUDS_INDEX} (Cloud Providers):
       - provider_name (keyword): Name of cloud provider
       - provider_name_text (text): Searchable provider name
       - overview (text): Provider overview
       - advantages (text): Provider advantages
       - weaknesses (text): Provider weaknesses  
       - pricing_structures (text): Pricing information
       - core_features (text): Key features
       - use_cases (text): Use case examples
       - success_stories (text): Success stories
       - analyst_opinions (text): Expert opinions
       - body_vector (dense_vector): 384-dim embeddings
    
    2. {BEST_PRACTICES_INDEX} (Best Practices):
       - id (integer): Practice ID
       - category (keyword): Practice category
       - title (text): Practice title
       - summary (text): Brief summary
       - detailed_description (text): Full description
       - impact_level (keyword): Impact assessment
       - implementation_effort (keyword): Effort required
       - examples (text): Implementation examples
       - body_vector (dense_vector): 384-dim embeddings
    """
    
    try:
        sql_prompt = f"""
        {str(system_prompt)}
        
        You are the SQL Query Tool (Step 2). Convert the clarified query into Elasticsearch query syntax.
        
        Database Schema:
        {str(schema_context)}
        
        User's Clarified Query: {str(request.clarified_query)}
        
        Generate an Elasticsearch query that will find the most relevant documents. 
        Focus on:
        - Text matching across relevant fields
        - Proper field targeting based on query content
        - Boosting important fields (title, provider_name, category)
        - Using appropriate query types (match, multi_match, bool)
        
        Return ONLY the Elasticsearch query as a JSON object, no explanations:
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an Elasticsearch query expert. Generate efficient, targeted queries."},
                {"role": "user", "content": sql_prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Generate explanation
        explanation_prompt = f"""
        Explain this Elasticsearch query in simple terms for: "{str(request.clarified_query)}"
        
        Query: {str(sql_query)}
        
        Provide a brief explanation of the search strategy:
        """
        
        explanation_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining database queries."},
                {"role": "user", "content": explanation_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        explanation = explanation_response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback to basic query
        sql_query = json.dumps({
            "query": {
                "multi_match": {
                    "query": request.clarified_query,
                    "fields": [
                        "provider_name^3",
                        "title^3", 
                        "category^2",
                        "overview^2",
                        "summary^2",
                        "advantages",
                        "core_features",
                        "detailed_description",
                        "use_cases"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": 50
        }, indent=2)
        
        explanation = f"""Multi-field search for: '{request.clarified_query}'
        - Searches across provider names, titles, and descriptions
        - Boosts matches in provider names and titles (higher relevance)
        - Uses fuzzy matching for typo tolerance
        - Returns top 50 results"""
    
    return SQLQueryResponse(
        sql_query=sql_query.strip(),
        explanation=explanation.strip(),
        estimated_relevance=0.85
    )

# ============================================================================
# Cloud Ratings SQL Query Generator
# ============================================================================

@app.post("/cloud-ratings-sql",
          response_model=CloudRatingsSQLResponse,
          summary="Cloud Ratings SQL Generator",
          description="Generates SQLite queries for the cloud ratings database")
async def cloud_ratings_sql_generator(request: CloudRatingsSQLRequest) -> CloudRatingsSQLResponse:
    """
    Generates SQLite queries specifically for the cloud ratings database.
    This endpoint understands the cloud_ratings table schema and can generate
    queries for analysis, comparisons, rankings, and specific provider lookups.
    """
    
    # Define the cloud ratings database schema
    schema_context = """
    Cloud Ratings Database Schema (SQLite):
    
    Table: cloud_ratings
    - id (INTEGER PRIMARY KEY): Auto-increment record ID
    - provider (TEXT UNIQUE): Cloud provider name (AWS, GCP, Azure, etc.)
    - ai_capabilities (REAL): AI capabilities score (0-10)
    - performance (REAL): Performance score (0-10)
    - cost_efficiency (REAL): Cost efficiency score (0-10)
    - flexibility (REAL): Flexibility score (0-10)
    - customer_service (REAL): Customer service score (0-10)
    - sustainability_score (REAL): Sustainability score (0-10)
    - ecosystem_innovation (REAL): Ecosystem innovation score (0-10)
    - data_sovereignty_strength (REAL): Data sovereignty score (0-10)
    - aggregate_score (REAL): Overall aggregate score (0-10)
    - scoring_version (TEXT): Version of scoring methodology
    - last_updated (DATE): Last update date
    - created_at (TIMESTAMP): Record creation timestamp
    - updated_at (TIMESTAMP): Record update timestamp
    
    Available View: cloud_ratings_summary
    - Includes all cloud_ratings columns plus ranking based on aggregate_score
    - Pre-sorted by aggregate_score DESC
    
    Sample queries can include:
    - Rankings and comparisons
    - Filtering by score thresholds
    - Finding best providers for specific criteria
    - Statistical analysis (averages, mins, maxs)
    - Provider-specific details
    """
    
    try:
        # Detect query type from the question
        question_lower = request.question.lower()
        detected_type = request.query_type
        
        if any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
            detected_type = "comparison"
        elif any(word in question_lower for word in ['top', 'best', 'ranking', 'rank', 'highest', 'lowest']):
            detected_type = "ranking"
        elif any(word in question_lower for word in ['aws', 'gcp', 'azure', 'ibm', 'alibaba', 'specific provider']):
            detected_type = "specific"
        else:
            detected_type = "analysis"
        
        # Generate SQL query using OpenAI
        sql_prompt = f"""
        You are a SQLite query expert for cloud provider ratings analysis.
        
        Database Schema:
        {schema_context}
        
        User Question: {request.question}
        Detected Query Type: {detected_type}
        
        Generate a SQLite query that answers the user's question. Guidelines:
        
        1. For RANKING queries: Use ORDER BY and LIMIT appropriately
        2. For COMPARISON queries: Compare specific providers or criteria
        3. For ANALYSIS queries: Use aggregation functions (AVG, MAX, MIN, COUNT)
        4. For SPECIFIC queries: Filter by provider names or specific criteria
        
        Important:
        - Use proper SQLite syntax
        - Include helpful column aliases
        - Use ROUND() for decimal places in calculations
        - Consider using the cloud_ratings_summary view for rankings
        - Return only the SQL query, no explanations
        
        Query:
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a SQLite expert specializing in cloud provider analysis queries. Generate clean, efficient SQL."},
                {"role": "user", "content": sql_prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the SQL query (remove markdown formatting if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Generate explanation
        explanation_prompt = f"""
        Explain this SQLite query for cloud ratings analysis in simple terms:
        
        Query: {sql_query}
        Original Question: {request.question}
        
        Provide a clear, concise explanation of:
        1. What data the query retrieves
        2. How it answers the user's question
        3. Any calculations or filtering involved
        
        Keep it under 150 words:
        """
        
        explanation_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining database queries clearly."},
                {"role": "user", "content": explanation_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        explanation = explanation_response.choices[0].message.content.strip()
        
        # Generate example usage
        example_usage = f"""
# Using Python with sqlite3
import sqlite3

conn = sqlite3.connect('cloud_ratings.db')
cursor = conn.cursor()

cursor.execute('''
{sql_query}
''')

results = cursor.fetchall()
for row in results:
    print(row)

conn.close()
        """.strip()
        
        # Estimate results description
        if 'top' in request.question.lower() or 'best' in request.question.lower():
            estimated_results = "A ranked list of cloud providers based on the specified criteria"
        elif 'compare' in request.question.lower():
            estimated_results = "Side-by-side comparison of specified cloud providers"
        elif any(provider in request.question.lower() for provider in ['aws', 'gcp', 'azure', 'ibm']):
            estimated_results = "Detailed information about the specified cloud provider(s)"
        else:
            estimated_results = "Analysis results based on the cloud ratings data"
        
        confidence_score = 0.9  # High confidence for structured data queries
        
    except Exception as e:
        # Fallback query generation
        if 'top' in request.question.lower() or 'best' in request.question.lower():
            sql_query = """
SELECT provider, aggregate_score, ranking
FROM cloud_ratings_summary
ORDER BY aggregate_score DESC
LIMIT 5;
            """.strip()
            explanation = "This query shows the top 5 cloud providers ranked by their aggregate scores."
            detected_type = "ranking"
        elif 'compare' in request.question.lower():
            sql_query = """
SELECT provider, ai_capabilities, performance, cost_efficiency, sustainability_score, aggregate_score
FROM cloud_ratings
WHERE provider IN ('AWS', 'GCP', 'Azure')
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = "This query compares key metrics between major cloud providers (AWS, GCP, Azure)."
            detected_type = "comparison"
        else:
            sql_query = """
SELECT provider, aggregate_score
FROM cloud_ratings
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = "This query lists all cloud providers with their aggregate scores in descending order."
            detected_type = "analysis"
        
        example_usage = f"""
# Using Python with sqlite3
import sqlite3

conn = sqlite3.connect('cloud_ratings.db')
cursor = conn.cursor()

cursor.execute('''
{sql_query}
''')

results = cursor.fetchall()
for row in results:
    print(row)

conn.close()
        """.strip()
        
        estimated_results = "Cloud provider data based on the generated query"
        confidence_score = 0.7
    
    return CloudRatingsSQLResponse(
        sql_query=sql_query,
        explanation=explanation,
        example_usage=example_usage,
        confidence_score=confidence_score,
        query_type=detected_type,
        estimated_results=estimated_results
    )

# ============================================================================
# Cloud Ratings SQL Execution Functions
# ============================================================================

def execute_cloud_ratings_sql(sql_query: str, db_path: str = None) -> Dict[str, Any]:
    """Execute a SQL query on the cloud ratings database and return results."""
    
    if db_path is None:
        # Check for database in multiple locations for containerization
        possible_paths = [
            os.getenv("DB_PATH", "/app/data/cloud_ratings.db"),  # Container path
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud_ratings.db'),  # Local dev
            './cloud_ratings.db'  # Fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        else:
            # If no database found, use the first path (container default)
            db_path = possible_paths[0]
    
    import time
    start_time = time.time()
    
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            return {
                "success": False,
                "error": f"Database not found at {db_path}. Please run create_cloud_ratings_database.py first.",
                "results": [],
                "column_names": [],
                "row_count": 0,
                "execution_time": 0.0
            }
        
        # Connect and execute query
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        # Convert rows to list of dictionaries
        results = []
        for row in rows:
            row_dict = {}
            for i, col_name in enumerate(column_names):
                value = row[i]
                # Convert any numeric values appropriately
                if isinstance(value, (int, float)):
                    row_dict[col_name] = value
                else:
                    row_dict[col_name] = str(value) if value is not None else None
            results.append(row_dict)
        
        conn.close()
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "results": results,
            "column_names": column_names,
            "row_count": len(results),
            "execution_time": execution_time,
            "error": None
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"SQL execution error: {str(e)}",
            "results": [],
            "column_names": [],
            "row_count": 0,
            "execution_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "results": [],
            "column_names": [],
            "row_count": 0,
            "execution_time": time.time() - start_time
        }

@app.post("/cloud-ratings-sql-execute",
          response_model=CloudRatingsSQLExecutionResponse,
          summary="Cloud Ratings SQL Executor",
          description="Generates and executes SQLite queries for cloud ratings database, returning actual results")
async def cloud_ratings_sql_executor(request: CloudRatingsSQLExecutionRequest) -> CloudRatingsSQLExecutionResponse:
    """
    Generates and executes SQLite queries for the cloud ratings database.
    This endpoint both generates the appropriate SQL and executes it,
    returning the actual data results for use in downstream processing.
    """
    
    # First, generate the SQL query using the existing logic
    sql_generation_request = CloudRatingsSQLRequest(
        question=request.question,
        query_type=request.query_type
    )
    
    sql_generation_response = await cloud_ratings_sql_generator(sql_generation_request)
    
    # If execution is disabled, return empty results
    if not request.execute_query:
        return CloudRatingsSQLExecutionResponse(
            sql_query=sql_generation_response.sql_query,
            query_results=[],
            column_names=[],
            row_count=0,
            execution_time=0.0,
            explanation=sql_generation_response.explanation,
            confidence_score=sql_generation_response.confidence_score,
            query_type=sql_generation_response.query_type
        )
    
    # Execute the generated SQL query
    execution_result = execute_cloud_ratings_sql(sql_generation_response.sql_query)
    
    if not execution_result["success"]:
        # If execution failed, still return the query but with error info
        return CloudRatingsSQLExecutionResponse(
            sql_query=sql_generation_response.sql_query,
            query_results=[{"error": execution_result["error"]}],
            column_names=["error"],
            row_count=0,
            execution_time=execution_result["execution_time"],
            explanation=f"Query generation succeeded but execution failed: {execution_result['error']}",
            confidence_score=0.3,  # Lower confidence due to execution failure
            query_type=sql_generation_response.query_type
        )
    
    return CloudRatingsSQLExecutionResponse(
        sql_query=sql_generation_response.sql_query,
        query_results=execution_result["results"],
        column_names=execution_result["column_names"],
        row_count=execution_result["row_count"],
        execution_time=execution_result["execution_time"],
        explanation=sql_generation_response.explanation,
        confidence_score=sql_generation_response.confidence_score,
        query_type=sql_generation_response.query_type
    )

# ============================================================================
# Endpoint 3: Vector DB Query Tool
# ============================================================================

@app.post("/vector-search",
          response_model=VectorSearchResponse,
          summary="Vector Database Search",
          description="Performs similarity search on vector database using clarified query and SQL results")
async def vector_search(request: VectorSearchRequest) -> VectorSearchResponse:
    """
    The Vector DB Query Tool performs semantic similarity search on the vector
    database using the user's clarified query and optionally incorporating
    results from the SQL query for enhanced context.
    """
    
    import time
    from datetime import datetime
    
    start_time = time.time()
    
    # Check if Elasticsearch client is available
    if es is None:
        return VectorSearchResponse(
            results=[],
            total_found=0,
            search_metadata={
                "error": "Elasticsearch client not available",
                "search_type": "unavailable",
                "message": "Vector search requires Elasticsearch connection"
            }
        )
    
    try:
        # Generate query embedding via configured backend
        embedding_start = time.time()
        dims = get_dense_vector_dims(es, CLOUDS_INDEX, "body_vector")
        query_vector = get_query_embedding(request.clarified_query, es_client=es, expected_dims=dims)
        embedding_time = time.time() - embedding_start
        
        # Search both indices with vector similarity
        search_start = time.time()
        
        # Define vector search query
        vector_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'body_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": request.max_results,
            "_source": {"excludes": ["body_vector"]}  # Exclude vector from results for performance
        }
        
        # Search cloud providers index
        try:
            clouds_response = es.search(index=CLOUDS_INDEX, **vector_query)
        except Exception as e_clouds:
            print(f"Failed to search clouds index: {e_clouds}")
            clouds_response = {"hits": {"hits": [], "total": {"value": 0}}}
        
        # Search best practices index  
        try:
            practices_response = es.search(index=BEST_PRACTICES_INDEX, **vector_query)
        except Exception as e_practices:
            print(f"Failed to search practices index: {e_practices}")
            practices_response = {"hits": {"hits": [], "total": {"value": 0}}}
        
        search_time = time.time() - search_start
        
        # Process and combine results
        all_results = []
        
        # Process cloud provider results
        for hit in clouds_response['hits']['hits']:
            result = {
                "document_id": hit['_id'],
                "index": CLOUDS_INDEX,
                "similarity_score": round(hit['_score'] - 1.0, 4),  # Subtract 1.0 added in script
                "title": hit['_source'].get('provider_name', 'Unknown Provider'),
                "content_snippet": hit['_source'].get('overview', '')[:300] + '...' if len(hit['_source'].get('overview', '')) > 300 else hit['_source'].get('overview', ''),
                "metadata": {
                    "type": "cloud_provider",
                    "provider_name": hit['_source'].get('provider_name'),
                    "advantages": hit['_source'].get('advantages', []),
                    "weaknesses": hit['_source'].get('weaknesses', []),
                    "core_features": hit['_source'].get('core_features', []),
                    "use_cases": hit['_source'].get('use_cases', [])
                }
            }
            all_results.append(result)
        
        # Process best practices results
        for hit in practices_response['hits']['hits']:
            result = {
                "document_id": hit['_id'],
                "index": BEST_PRACTICES_INDEX,
                "similarity_score": round(hit['_score'] - 1.0, 4),  # Subtract 1.0 added in script
                "title": hit['_source'].get('title', 'Unknown Practice'),
                "content_snippet": hit['_source'].get('summary', '')[:300] + '...' if len(hit['_source'].get('summary', '')) > 300 else hit['_source'].get('summary', ''),
                "metadata": {
                    "type": "best_practice",
                    "category": hit['_source'].get('category'),
                    "impact_level": hit['_source'].get('impact_level'),
                    "implementation_effort": hit['_source'].get('implementation_effort'),
                    "examples": hit['_source'].get('examples', []),
                    "detailed_description": hit['_source'].get('detailed_description', '')
                }
            }
            all_results.append(result)
        
        # Sort all results by similarity score (descending)
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit to max_results
        final_results = all_results[:request.max_results]
        
        # Filter based on SQL results if provided
        if request.sql_results:
            # Extract IDs or criteria from SQL results for filtering
            sql_ids = set()
            for sql_result in request.sql_results:
                if 'id' in sql_result:
                    sql_ids.add(str(sql_result['id']))
                elif 'provider_name' in sql_result:
                    sql_ids.add(sql_result['provider_name'])
            
            if sql_ids:
                # Filter results to only include those matching SQL criteria
                filtered_results = []
                for result in final_results:
                    result_id = result['document_id']
                    provider_name = result['metadata'].get('provider_name', '')
                    if result_id in sql_ids or provider_name in sql_ids:
                        filtered_results.append(result)
                final_results = filtered_results
        
        total_time = time.time() - start_time
        
        search_metadata = {
            "query_embedding_time": f"{embedding_time:.3f}s",
            "vector_search_time": f"{search_time:.3f}s",
            "total_processing_time": f"{total_time:.3f}s",
            "total_documents_searched": clouds_response['hits']['total']['value'] + practices_response['hits']['total']['value'],
            "clouds_hits": clouds_response['hits']['total']['value'],
            "practices_hits": practices_response['hits']['total']['value'],
            "search_algorithm": "cosine_similarity",
            "indices_searched": [CLOUDS_INDEX, BEST_PRACTICES_INDEX],
            "query_timestamp": datetime.now().isoformat()
        }
        
        final_results = final_results
        total_found = len(final_results)
        
    except Exception as e:
        # Fallback to basic search if vector search fails
        print(f"Vector search error: {e}")
        
        # Basic text search fallback
        text_query = {
            "query": {
                "multi_match": {
                    "query": request.clarified_query,
                    "fields": ["provider_name^2", "title^2", "overview", "summary", "advantages", "core_features"],
                    "fuzziness": "AUTO"
                }
            },
            "size": request.max_results
        }
        
        try:
            clouds_response = es.search(index=CLOUDS_INDEX, **text_query)
            practices_response = es.search(index=BEST_PRACTICES_INDEX, **text_query)
            
            final_results = []
            for hit in clouds_response['hits']['hits'] + practices_response['hits']['hits']:
                result = {
                    "document_id": hit['_id'],
                    "similarity_score": hit['_score'] / 10.0,  # Normalize text score
                    "title": hit['_source'].get('provider_name') or hit['_source'].get('title', 'Unknown'),
                    "content_snippet": (hit['_source'].get('overview') or hit['_source'].get('summary', ''))[:300],
                    "metadata": {"type": "text_search_fallback", "source": hit['_source']}
                }
                final_results.append(result)
            
            final_results = final_results[:request.max_results]
            total_found = len(final_results)
            search_metadata = {"search_type": "text_fallback", "error": str(e)}
            
        except Exception as fallback_error:
            # Ultimate fallback
            final_results = []
            total_found = 0
            search_metadata = {"error": f"Search failed: {e}, Fallback failed: {fallback_error}"}
    
    return VectorSearchResponse(
        results=final_results,
        total_found=total_found,
        search_metadata=search_metadata
    )

# ============================================================================
# Endpoint 4: Cohesive Answer Tool
# ============================================================================

@app.post("/test-answer", summary="Test Answer Generator")
async def test_answer_generator(request: CohesiveAnswerRequest) -> CohesiveAnswerResponse:
    """Simple test version of the answer generator"""
    try:
        # Very simple synthesis without complex OpenAI calls
        answer = f"Based on your query '{str(request.original_query)}', I found {str(len(request.vector_results))} relevant results."
        
        sources = []
        for result in request.vector_results[:3]:  # Limit to first 3
            sources.append({
                "title": str(result.get("title", "Unknown")),
                "type": "test",
                "similarity_score": str(result.get("similarity_score", 0)),
                "url": "#test"
            })
        
        return CohesiveAnswerResponse(
            answer=answer,
            sources=sources,
            confidence_score=0.75,
            additional_recommendations=["Test recommendation 1", "Test recommendation 2"]
        )
    except Exception as e:
        # Return error details for debugging
        return CohesiveAnswerResponse(
            answer=f"Error occurred: {str(e)}",
            sources=[],
            confidence_score=0.0,
            additional_recommendations=[]
        )

@app.post("/cohesive-answer",
          response_model=CohesiveAnswerResponse,
          summary="Cohesive Answer Generator",
          description="Generates comprehensive answers from vector search results")
async def cohesive_answer_generator(request: CohesiveAnswerRequest) -> CohesiveAnswerResponse:
    """
    The Cohesive Answer Tool synthesizes findings from the vector database
    search into a comprehensive, well-structured answer to the user's
    original research question.
    """
    
    try:
        # Extract key information from vector search results
        cloud_providers = []
        best_practices = []
        sources = []
        
        for result in request.vector_results:
            metadata = result.get("metadata", {})
            source = {
                "title": result.get("title", "Unknown Title"),
                "type": metadata.get("type", "unknown"),
                "similarity_score": str(result.get("similarity_score", 0)),
                "url": f"#{result.get('document_id', 'unknown')}"
            }
            sources.append(source)
            
            if metadata.get("type") == "cloud_provider":
                cloud_providers.append(result)
            elif metadata.get("type") == "best_practice":
                best_practices.append(result)
        
        # Prepare context for OpenAI - including both vector and SQL results
        results_context = ""
        
        # Add vector search results
        for i, result in enumerate(request.vector_results[:10]):  # Limit to top 10 for context
            similarity_score = float(result.get('similarity_score', 0))
            results_context += f"""
            Vector Result {str(i+1)} (Score: {similarity_score:.3f}):
            Title: {str(result.get('title', 'Unknown'))}
            Type: {str(result.get('metadata', {}).get('type', 'unknown'))}
            Content: {str(result.get('content_snippet', ''))}
            
            """
        
        # Add SQL context if available
        sql_context = ""
        if request.sql_context and len(request.sql_context) > 0:
            sql_context = f"""
            
            STRUCTURED DATA INSIGHTS (from Cloud Ratings Database):
            Found {len(request.sql_context)} relevant records:
            
            """
            
            for i, sql_result in enumerate(request.sql_context[:10]):  # Limit to 10 for context
                sql_context += f"""
                SQL Record {str(i+1)}:
                """
                for key, value in sql_result.items():
                    if key != 'id':  # Skip internal ID fields
                        sql_context += f"  {str(key).replace('_', ' ').title()}: {str(value)}\n"
                sql_context += "\n"
        
        results_context += sql_context
        
        # Generate comprehensive answer and recommendations in one OpenAI call
        synthesis_prompt = f"""
        {str(system_prompt)}
        
        You are the Cohesive Answer Tool (Step 4). Synthesize the research findings into a clear, 
        accurate, actionable recommendation with best practices and next steps.
        
        Original Query: {str(request.original_query)}
        
        Available Data Sources:
        - Vector Search Results: {str(len(request.vector_results))} semantic matches from knowledge base
        - Structured Data: {str(len(request.sql_context))} records from cloud ratings database
        
        Combined Research Context:
        {str(results_context)}
        
        Generate a comprehensive response with TWO sections:
        
        1. ANSWER: A comprehensive analysis that:
           - Directly addresses the user's question
           - Synthesizes key findings from BOTH vector search results AND structured data
           - Leverages specific metrics and scores from the cloud ratings database when available
           - Provides actionable recommendations based on concrete data
           - Identifies best practices when relevant
           - Notes any limitations or considerations
           - Highlights quantitative insights from the structured data where applicable
        
        2. RECOMMENDATIONS: 3-5 specific, actionable follow-up recommendations for further research or implementation.
        
        Format your response as:
        ANSWER:
        [your comprehensive answer here]
        
        RECOMMENDATIONS:
        - [recommendation 1]
        - [recommendation 2]
        - [recommendation 3]
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a research synthesis expert. Create comprehensive, actionable answers from search results."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Parse the response to extract answer and recommendations
        if "RECOMMENDATIONS:" in full_response:
            answer, recs_section = full_response.split("RECOMMENDATIONS:", 1)
            answer = answer.replace("ANSWER:", "").strip()
            recommendations = [line.strip().lstrip("- ") for line in recs_section.split('\n') if line.strip() and line.strip().startswith('-')]
        else:
            answer = full_response.replace("ANSWER:", "").strip()
            recommendations = ["Review the search results for detailed guidance", "Consider the specific context of your requirements"]
        
        # Calculate confidence score based on result quality
        avg_similarity = sum(r.get('similarity_score', 0) for r in request.vector_results) / max(len(request.vector_results), 1)
        result_count_factor = min(len(request.vector_results) / 10, 1.0)  # More results = higher confidence
        confidence_score = min((avg_similarity * 0.7 + result_count_factor * 0.3), 0.95)
        
    except Exception as e:
        # Log the actual error for debugging
        print(f"ERROR in cohesive_answer_generator: {str(e)}")
        print(f"ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback synthesis
        sources = []
        for result in request.vector_results:
            source = {
                "title": result.get("title", "Unknown Title"),
                "type": result.get("metadata", {}).get("type", "unknown"),
                "similarity_score": str(result.get("similarity_score", 0)),
                "url": f"#{result.get('document_id', 'unknown')}"
            }
            sources.append(source)
        
        answer = f"""
        Based on the research findings for your query: "{request.original_query}"

        Analysis Summary:
        The search identified {len(request.vector_results)} relevant documents that address your question. 
        The findings include information from cloud providers and best practices documentation.

        Key Insights:
        - Found {len([r for r in request.vector_results if r.get('metadata', {}).get('type') == 'cloud_provider'])} cloud provider resources
        - Identified {len([r for r in request.vector_results if r.get('metadata', {}).get('type') == 'best_practice'])} best practice documents
        - Average similarity score: {sum(r.get('similarity_score', 0) for r in request.vector_results) / max(len(request.vector_results), 1):.3f}

        This analysis is based on {len(sources)} sources with semantic similarity to your research question.
        """
        
        recommendations = [
            "Review the highest-scoring results for detailed implementation guidance",
            "Consider the specific use cases and requirements mentioned in the findings",
            "Evaluate the trade-offs between different approaches identified in the search"
        ]
        
        confidence_score = 0.65
    
    return CohesiveAnswerResponse(
        answer=answer.strip(),
        sources=sources,
        confidence_score=confidence_score,
        additional_recommendations=recommendations
    )

# ============================================================================
# Combined Research Pipeline Endpoint
# ============================================================================

class CombinedResearchRequest(BaseModel):
    """Request model for combined research pipeline"""
    question: str = Field(..., description="The user's research question")
    max_clarification_questions: Optional[int] = Field(default=2, description="Maximum clarification questions to ask (0-3)")
    max_results: Optional[int] = Field(default=10, description="Maximum number of vector search results")
    skip_clarification: Optional[bool] = Field(default=False, description="Skip clarification step and proceed directly")

class CombinedResearchResponse(BaseModel):
    """Response model for combined research pipeline"""
    original_question: str = Field(..., description="The original user question")
    clarified_query: Optional[str] = Field(None, description="The clarified query after processing")
    final_answer: str = Field(..., description="The comprehensive final answer")
    sources: List[Dict[str, str]] = Field(..., description="Sources and references used")
    confidence_score: float = Field(..., description="Overall confidence in the answer (0-1)")
    additional_recommendations: List[str] = Field(..., description="Additional research suggestions")
    processing_metadata: Dict[str, Any] = Field(..., description="Metadata about the processing pipeline")

@app.post("/research-pipeline",
          response_model=CombinedResearchResponse,
          summary="Combined Research Pipeline",
          description="Complete research pipeline: takes a question and returns a comprehensive answer")
async def combined_research_pipeline(request: CombinedResearchRequest) -> CombinedResearchResponse:
    """
    Combined Research Pipeline that orchestrates all four tools:
    1. Clarification Agent (optional, configurable)
    2. SQL Query Generator 
    3. Vector Database Search
    4. Cohesive Answer Generator
    
    This endpoint provides a single entry point for the complete research workflow.
    """
    
    import time
    start_time = time.time()
    processing_steps = []
    
    try:
        # Step 1: Clarification (optional)
        clarified_query = request.question
        clarification_info = {"skipped": request.skip_clarification}
        
        if not request.skip_clarification and request.max_clarification_questions > 0:
            processing_steps.append("clarification_processing")
            
            # For the combined endpoint, we'll do a simplified clarification
            # that generates a better query without interactive back-and-forth
            try:
                clarification_prompt = f"""
                {str(system_prompt)}
                
                You are a research query optimizer. Enhance this user question to be more specific and searchable:
                
                Original Question: {str(request.question)}
                
                Generate an enhanced, more specific version that:
                - Maintains the user's intent
                - Adds relevant technical context
                - Makes it more searchable for cloud and best practices databases
                - Is clear and comprehensive
                
                Return only the enhanced query, no explanations:
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a research query optimization expert."},
                        {"role": "user", "content": clarification_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                clarified_query = response.choices[0].message.content.strip()
                clarification_info = {"enhanced": True, "method": "automated_enhancement"}
                
            except Exception as e:
                # Fallback to original query
                clarified_query = request.question
                clarification_info = {"enhanced": False, "error": str(e), "fallback": True}
        
        # Step 2A: Elasticsearch SQL Query Generation (for vector search context)
        processing_steps.append("elasticsearch_sql_generation")
        sql_request = SQLQueryRequest(clarified_query=clarified_query)
        elasticsearch_sql_response = await sql_query_generator(sql_request)
        
        # Step 2B: Cloud Ratings SQL Execution (for structured data insights)
        processing_steps.append("cloud_ratings_sql_execution")
        cloud_sql_results = []
        cloud_sql_metadata = {}
        
        try:
            # Check if the query might benefit from cloud ratings data
            query_lower = clarified_query.lower()
            if any(keyword in query_lower for keyword in [
                'cloud', 'provider', 'aws', 'gcp', 'azure', 'ibm', 'alibaba',
                'performance', 'cost', 'efficiency', 'ai', 'sustainability',
                'rating', 'score', 'compare', 'best', 'top', 'ranking'
            ]):
                cloud_sql_request = CloudRatingsSQLExecutionRequest(
                    question=clarified_query,
                    execute_query=True,
                    query_type="analysis"
                )
                cloud_sql_response = await cloud_ratings_sql_executor(cloud_sql_request)
                cloud_sql_results = cloud_sql_response.query_results
                cloud_sql_metadata = {
                    "sql_query": cloud_sql_response.sql_query,
                    "row_count": cloud_sql_response.row_count,
                    "execution_time": cloud_sql_response.execution_time,
                    "query_type": cloud_sql_response.query_type,
                    "confidence_score": cloud_sql_response.confidence_score
                }
            else:
                cloud_sql_metadata = {"skipped": True, "reason": "Query does not appear to need cloud ratings data"}
                
        except Exception as e:
            cloud_sql_metadata = {"error": str(e), "fallback": True}
        
        # Step 3: Enhanced Vector Search (with SQL context)
        processing_steps.append("enhanced_vector_search")
        vector_request = VectorSearchRequest(
            clarified_query=clarified_query,
            sql_results=cloud_sql_results,  # Pass actual SQL results for enhanced filtering
            max_results=request.max_results
        )
        vector_response = await vector_search(vector_request)
        
        # Step 4: Enhanced Cohesive Answer Generation (with SQL insights)
        processing_steps.append("enhanced_answer_synthesis")
        answer_request = CohesiveAnswerRequest(
            original_query=request.question,
            vector_results=vector_response.results,
            sql_context=cloud_sql_results  # Pass SQL results for comprehensive analysis
        )
        answer_response = await cohesive_answer_generator(answer_request)
        
        total_time = time.time() - start_time
        
        # Compile processing metadata
        processing_metadata = {
            "total_processing_time": f"{total_time:.3f}s",
            "steps_completed": processing_steps,
            "clarification_info": clarification_info,
            "elasticsearch_sql_query_generated": bool(elasticsearch_sql_response.sql_query),
            "cloud_ratings_sql_metadata": cloud_sql_metadata,
            "cloud_ratings_results_count": len(cloud_sql_results),
            "vector_results_found": vector_response.total_found,
            "vector_search_metadata": vector_response.search_metadata,
            "pipeline_version": "2.0.0",  # Updated version with SQL execution
            "timestamp": datetime.now().isoformat()
        }
        
        return CombinedResearchResponse(
            original_question=request.question,
            clarified_query=clarified_query,
            final_answer=answer_response.answer,
            sources=answer_response.sources,
            confidence_score=answer_response.confidence_score,
            additional_recommendations=answer_response.additional_recommendations,
            processing_metadata=processing_metadata
        )
        
    except Exception as e:
        # Comprehensive error handling with fallback response
        total_time = time.time() - start_time
        
        processing_metadata = {
            "total_processing_time": f"{total_time:.3f}s",
            "steps_completed": processing_steps,
            "error": str(e),
            "error_step": processing_steps[-1] if processing_steps else "initialization",
            "pipeline_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate a basic fallback answer
        fallback_answer = f"""
        I encountered an issue while processing your research question: "{request.question}"
        
        Error occurred during: {processing_steps[-1] if processing_steps else 'initialization'}
        
        While I cannot provide the full research pipeline results at this time, here are some general suggestions:
        
        1. Please ensure your question is clear and specific
        2. Consider breaking down complex queries into simpler parts
        3. Try rephrasing your question if it's very technical
        
        Error details: {str(e)[:200]}...
        """
        
        return CombinedResearchResponse(
            original_question=request.question,
            clarified_query=request.question,
            final_answer=fallback_answer,
            sources=[],
            confidence_score=0.1,
            additional_recommendations=[
                "Try simplifying your question",
                "Check if the research service is properly configured",
                "Contact support if the issue persists"
            ],
            processing_metadata=processing_metadata
        )

# ============================================================================
# Health Check and Utility Endpoints
# ============================================================================

@app.get("/health", summary="Health Check", description="Check if the service is running")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/", summary="Service Information", description="Get information about the service")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Research Query Processing Tools",
        "version": "1.0.0",
        "description": "AI-powered tools for watsonx Orchestrate",
        "endpoints": [
            "/clarification-agent",
            "/sql-query-generator", 
            "/cloud-ratings-sql",
            "/cloud-ratings-sql-execute",
            "/vector-search",
            "/cohesive-answer",
            "/research-pipeline"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Use PORT environment variable for Code Engine compatibility
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

