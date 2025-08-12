"""
FastAPI application for watsonx Orchestrate Agent Tools
=====================================================

This application provides four key endpoints designed to work as tools
within IBM watsonx Orchestrate Agent Development Kit (ADK):

1. Clarification Agent - Improves user queries by correcting grammar and adding context
2. SQL Query Tool - Converts natural language to SQL queries
3. Vector DB Query Tool - Performs similarity search on vector database
4. Cohesive Answer Tool - Generates final comprehensive responses

Each endpoint is designed to be imported as an OpenAPI-based tool in watsonx Orchestrate.
All endpoints are stateless and designed for single-turn operations.
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
INNOVATIVE_IDEAS_INDEX = "innovative_ideas"
TECH_TRENDS_INDEX = "tech_trends_innovation"

# Model configuration (for reference only when using local ST)
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Cloud ratings SQL generation will use fallback queries.")
    openai_client = None
else:
    # Initialize OpenAI client
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

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

class ClarificationRequest(BaseModel):
    """Request model for clarification agent"""
    user_query: str = Field(..., description="The user's query that needs grammatical improvement and context enhancement")

class ClarificationResponse(BaseModel):
    """Response model for clarification agent"""
    clarified_query: str = Field(..., description="Grammatically improved and contextually enhanced query")
    improvements_made: List[str] = Field(..., description="List of specific improvements made to the original query")
    confidence_score: float = Field(..., description="Confidence in the clarification quality (0-1)")



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

class CohesiveAnswerAutoRequest(BaseModel):
    """Request model for auto cohesive answer generation"""
    query: str = Field(..., description="The user's research question")
    max_results: Optional[int] = Field(default=10, description="Maximum number of vector search results to retrieve")

class CloudRatingsSQLRequest(BaseModel):
    """Request model for cloud ratings SQL query generation and execution"""
    question: str = Field(..., description="Natural language question about cloud ratings data")
    query_type: Optional[str] = Field(default="analysis", description="Type of query: analysis, comparison, ranking, or specific")
    execute_query: Optional[bool] = Field(default=True, description="Whether to execute the generated query and return results")

class CloudRatingsSQLResponse(BaseModel):
    """Response model for cloud ratings SQL query generation and optional execution"""
    sql_query: str = Field(..., description="Generated SQLite query for cloud ratings database")
    explanation: str = Field(..., description="Explanation of what the query does")
    example_usage: str = Field(..., description="Example of how to execute the query")
    confidence_score: float = Field(..., description="Confidence in query accuracy (0-1)")
    query_type: str = Field(..., description="Detected type of query")
    estimated_results: str = Field(..., description="Description of expected results")
    # Optional execution results
    query_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Actual query results if executed")
    column_names: Optional[List[str]] = Field(default=None, description="Column names if query was executed")
    row_count: Optional[int] = Field(default=None, description="Number of rows returned if executed")
    execution_time: Optional[float] = Field(default=None, description="Query execution time in seconds if executed")
    execution_success: Optional[bool] = Field(default=None, description="Whether query execution was successful")



# ============================================================================
# Endpoint 1: Clarification Agent
# ============================================================================

@app.post("/clarification-agent", 
          response_model=ClarificationResponse,
          summary="Clarification Agent",
          description="Improves user queries by correcting grammar and adding relevant context")
async def clarification_agent(request: ClarificationRequest) -> ClarificationResponse:
    """
    The Clarification Agent improves user queries by:
    1. Correcting grammatical errors and improving clarity
    2. Adding relevant technical context and industry terminology
    3. Expanding abbreviations and clarifying ambiguous terms
    4. Structuring the query for better search results
    
    This is a single-turn operation designed for stateless integration.
    """
    
    try:
        clarification_prompt = f"""
        {str(system_prompt)}
        
        You are a Query Clarification Expert. Your task is to improve the user's query by:
        
        1. GRAMMAR & CLARITY: Fix any grammatical errors, typos, or unclear phrasing
        2. CONTEXT ENHANCEMENT: Add relevant technical context, industry terminology, and specificity
        3. STRUCTURE: Organize the query in a clear, logical manner
        4. COMPLETENESS: Expand abbreviations and clarify ambiguous terms
        
        Original Query: "{str(request.user_query)}"
        
        Provide:
        1. An improved, clarified version of the query
        2. A brief explanation of the improvements made
        
        Focus on making the query more precise, searchable, and comprehensive while preserving the user's original intent.
        
        Return your response in this exact format:
        CLARIFIED_QUERY: [the improved query here]
        IMPROVEMENTS: [comma-separated list of improvements made]
        """
        
        if openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query improvement expert. Enhance user queries while preserving their original intent."},
                    {"role": "user", "content": clarification_prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            if "CLARIFIED_QUERY:" in result and "IMPROVEMENTS:" in result:
                parts = result.split("IMPROVEMENTS:")
                clarified_query = parts[0].replace("CLARIFIED_QUERY:", "").strip()
                improvements_text = parts[1].strip()
                improvements_made = [imp.strip() for imp in improvements_text.split(",") if imp.strip()]
                confidence_score = 0.9
            else:
                # Fallback parsing
                clarified_query = result
                improvements_made = ["Grammar and clarity improvements", "Added technical context"]
                confidence_score = 0.7
                
        else:
            # Fallback when OpenAI is not available
            clarified_query = f"Enhanced query: {request.user_query} (with improved technical terminology and context for better research results)"
            improvements_made = ["Added technical context", "Improved structure for better searchability"]
            confidence_score = 0.6
            
    except Exception as e:
        logger.error(f"Error in clarification agent: {str(e)}")
        # Fallback response
        clarified_query = f"Clarified research query: {request.user_query}"
        improvements_made = ["Basic grammatical improvements applied"]
        confidence_score = 0.5
    
    return ClarificationResponse(
        clarified_query=clarified_query,
        improvements_made=improvements_made,
        confidence_score=confidence_score
    )

# ============================================================================
# Cloud Ratings SQL Query Generator & Executor (Combined)
# ============================================================================

@app.post("/cloud-ratings-sql",
          response_model=CloudRatingsSQLResponse,
          summary="Cloud Ratings SQL Generator & Executor",
          description="Generates and optionally executes SQLite queries for the cloud ratings database")
async def cloud_ratings_sql_generator(request: CloudRatingsSQLRequest) -> CloudRatingsSQLResponse:
    """
    Generates SQLite queries specifically for the cloud ratings database.
    Optionally executes the query and returns results. Perfect for ReAct agents.
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
    - Includes all major cloud_ratings columns plus ranking based on aggregate_score
    - Columns: provider, aggregate_score, ranking, ai_capabilities, performance, cost_efficiency, 
      flexibility, customer_service, sustainability_score, ecosystem_innovation, 
      data_sovereignty_strength, scoring_version, last_updated
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
        
        # Generate SQL query using OpenAI (if available)
        if openai_client is not None:
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
            confidence_score = 0.9  # High confidence for OpenAI-generated queries
        else:
            # Fallback to predefined queries when OpenAI is not available
            sql_query, explanation, confidence_score = generate_fallback_query(request.question, detected_type)
        
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
    
    # Initialize optional execution fields
    query_results = None
    column_names = None
    row_count = None
    execution_time = None
    execution_success = None
    
    # Execute query if requested
    if request.execute_query:
        execution_result = execute_cloud_ratings_sql(sql_query)
        execution_success = execution_result["success"]
        execution_time = execution_result["execution_time"]
        
        if execution_success:
            query_results = execution_result["results"]
            column_names = execution_result["column_names"]
            row_count = execution_result["row_count"]
        else:
            # Include error information in results
            query_results = [{"error": execution_result["error"]}]
            column_names = ["error"]
            row_count = 0
            # Adjust explanation to include execution error
            explanation += f"\n\nExecution Error: {execution_result['error']}"
    
    return CloudRatingsSQLResponse(
        sql_query=sql_query,
        explanation=explanation,
        example_usage=example_usage,
        confidence_score=confidence_score,
        query_type=detected_type,
        estimated_results=estimated_results,
        query_results=query_results,
        column_names=column_names,
        row_count=row_count,
        execution_time=execution_time,
        execution_success=execution_success
    )

# ============================================================================
# Cloud Ratings SQL Helper Functions
# ============================================================================

def generate_fallback_query(question: str, query_type: str) -> tuple[str, str, float]:
    """
    Generate fallback SQL queries when OpenAI is not available.
    
    Returns:
        tuple: (sql_query, explanation, confidence_score)
    """
    question_lower = question.lower()
    
    if query_type == "ranking" or any(word in question_lower for word in ['top', 'best', 'ranking', 'rank', 'highest']):
        # Extract number if mentioned (e.g., "top 5", "best 3")
        import re
        number_match = re.search(r'\b(\d+)\b', question)
        limit = number_match.group(1) if number_match else "5"
        
        sql_query = f"""
SELECT provider, aggregate_score, ranking,
       ai_capabilities, performance, cost_efficiency, sustainability_score
FROM cloud_ratings_summary
ORDER BY aggregate_score DESC
LIMIT {limit};
        """.strip()
        
        explanation = f"This query retrieves the top {limit} cloud providers ranked by their aggregate scores, including detailed metrics for AI capabilities, performance, cost efficiency, and sustainability."
        confidence_score = 0.8
        
    elif query_type == "comparison" or any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
        # Try to extract provider names
        providers = []
        provider_keywords = ['aws', 'azure', 'gcp', 'google cloud', 'ibm', 'oracle', 'alibaba']
        for keyword in provider_keywords:
            if keyword in question_lower:
                if keyword == 'gcp' or keyword == 'google cloud':
                    providers.append('GCP')
                elif keyword == 'aws':
                    providers.append('AWS')
                elif keyword == 'azure':
                    providers.append('Azure')
                elif keyword == 'ibm':
                    providers.append('IBM Cloud')
                elif keyword == 'oracle':
                    providers.append('Oracle Cloud')
                elif keyword == 'alibaba':
                    providers.append('Alibaba Cloud')
        
        if providers:
            provider_list = "', '".join(providers)
            sql_query = f"""
SELECT provider, aggregate_score, ai_capabilities, performance, 
       cost_efficiency, flexibility, customer_service, sustainability_score
FROM cloud_ratings
WHERE provider IN ('{provider_list}')
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = f"This query compares {', '.join(providers)} across all major rating categories including AI capabilities, performance, cost efficiency, and sustainability."
        else:
            sql_query = """
SELECT provider, aggregate_score, ai_capabilities, performance, 
       cost_efficiency, flexibility, customer_service, sustainability_score
FROM cloud_ratings
WHERE provider IN ('AWS', 'GCP', 'Azure')
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = "This query compares the top three major cloud providers (AWS, GCP, Azure) across all rating categories."
        
        confidence_score = 0.7
        
    elif query_type == "specific" or any(provider in question_lower for provider in ['aws', 'gcp', 'azure', 'ibm']):
        # Specific provider query
        provider = None
        if 'aws' in question_lower:
            provider = 'AWS'
        elif 'gcp' in question_lower or 'google' in question_lower:
            provider = 'GCP'
        elif 'azure' in question_lower:
            provider = 'Azure'
        elif 'ibm' in question_lower:
            provider = 'IBM Cloud'
        
        if provider:
            sql_query = f"""
SELECT provider, aggregate_score, ranking, ai_capabilities, performance,
       cost_efficiency, flexibility, customer_service, sustainability_score,
       ecosystem_innovation, data_sovereignty_strength, last_updated
FROM cloud_ratings_summary
WHERE provider = '{provider}';
            """.strip()
            explanation = f"This query retrieves comprehensive rating information for {provider}, including its ranking and scores across all evaluation categories."
        else:
            sql_query = """
SELECT provider, aggregate_score, ranking
FROM cloud_ratings_summary
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = "This query shows all cloud providers with their aggregate scores and rankings."
        
        confidence_score = 0.75
        
    else:
        # General analysis
        if any(word in question_lower for word in ['average', 'mean', 'avg']):
            sql_query = """
SELECT 
    ROUND(AVG(aggregate_score), 2) as avg_aggregate_score,
    ROUND(AVG(ai_capabilities), 2) as avg_ai_capabilities,
    ROUND(AVG(performance), 2) as avg_performance,
    ROUND(AVG(cost_efficiency), 2) as avg_cost_efficiency,
    ROUND(AVG(sustainability_score), 2) as avg_sustainability,
    COUNT(*) as total_providers
FROM cloud_ratings;
            """.strip()
            explanation = "This query calculates average scores across all cloud providers for each rating category, providing an industry overview."
        elif any(word in question_lower for word in ['count', 'how many', 'number']):
            sql_query = """
SELECT COUNT(*) as total_providers,
       COUNT(CASE WHEN aggregate_score >= 8.0 THEN 1 END) as high_rated_providers,
       COUNT(CASE WHEN aggregate_score < 8.0 THEN 1 END) as lower_rated_providers
FROM cloud_ratings;
            """.strip()
            explanation = "This query counts the total number of cloud providers and categorizes them by their aggregate rating scores."
        else:
            sql_query = """
SELECT provider, aggregate_score, ranking
FROM cloud_ratings_summary
ORDER BY aggregate_score DESC;
            """.strip()
            explanation = "This query lists all cloud providers with their aggregate scores and rankings in descending order."
        
        confidence_score = 0.6
    
    return sql_query, explanation, confidence_score


# ============================================================================
# Cloud Ratings SQL Execution Functions
# ============================================================================

def execute_cloud_ratings_sql(sql_query: str, db_path: str = None) -> Dict[str, Any]:
    """Execute a SQL query on the cloud ratings database and return results."""
    
    if db_path is None:
        # Check for database in multiple locations for containerization
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        possible_paths = [
            os.getenv("DB_PATH", "/app/data/cloud_ratings.db"),  # Container path
            os.path.join(script_dir, 'cloud_ratings.db'),  # Local dev
            './cloud_ratings.db'  # Fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        else:
            # If no database found, use the local fallback
            db_path = './cloud_ratings.db'
    
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
        logger.info(f"Vector search starting for query: '{request.clarified_query}'")
        
        # Determine vector field based on embedding mode
        vector_field = "body_vector"  # Default for sentence-transformers
        if EMBEDDINGS_MODE == "openai":
            vector_field = "body_vector_v2"
        
        logger.info(f"Using vector field: {vector_field}, embeddings mode: {EMBEDDINGS_MODE}")
        
        # Generate query embedding via configured backend
        embedding_start = time.time()
        dims = get_dense_vector_dims(es, CLOUDS_INDEX, vector_field)
        query_vector = get_query_embedding(request.clarified_query, es_client=es, expected_dims=dims)
        embedding_time = time.time() - embedding_start
        
        logger.info(f"Query embedding generated: dims={dims}, vector_length={len(query_vector) if query_vector else 0}")
        
        # Search both indices with vector similarity
        search_start = time.time()
        
        # Define vector search query
        vector_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": request.max_results,
            "_source": {"excludes": [vector_field]}  # Exclude vector from results for performance
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
        
        # Search innovative ideas index
        try:
            ideas_response = es.search(index=INNOVATIVE_IDEAS_INDEX, **vector_query)
        except Exception as e_ideas:
            print(f"Failed to search innovative ideas index: {e_ideas}")
            ideas_response = {"hits": {"hits": [], "total": {"value": 0}}}
        
        # Search tech trends index
        try:
            trends_response = es.search(index=TECH_TRENDS_INDEX, **vector_query)
        except Exception as e_trends:
            print(f"Failed to search tech trends index: {e_trends}")
            trends_response = {"hits": {"hits": [], "total": {"value": 0}}}
        
        search_time = time.time() - search_start
        
        # Process and combine results
        all_results = []
        
        logger.info(f"Vector search results summary:")
        logger.info(f"  - Clouds hits: {len(clouds_response['hits']['hits'])}")
        logger.info(f"  - Practices hits: {len(practices_response['hits']['hits'])}")
        logger.info(f"  - Ideas hits: {len(ideas_response['hits']['hits'])}")
        logger.info(f"  - Trends hits: {len(trends_response['hits']['hits'])}")
        
        # Process cloud provider results
        for i, hit in enumerate(clouds_response['hits']['hits']):
            logger.info(f"Cloud result {i}: id={hit['_id']}, score={hit['_score']}, source_keys={list(hit['_source'].keys())}")
            logger.info(f"Cloud result {i}: raw source sample: {str(hit['_source'])[:200]}...")
            
            # Robust field extraction - try multiple possible field names
            title = (hit['_source'].get('provider_name') or 
                    hit['_source'].get('name') or 
                    hit['_source'].get('title') or 
                    'Unknown Provider')
            
            content = (hit['_source'].get('overview') or 
                      hit['_source'].get('description') or 
                      hit['_source'].get('summary') or 
                      '')
            
            result = {
                "document_id": hit['_id'],
                "index": CLOUDS_INDEX,
                "similarity_score": round(hit['_score'] - 1.0, 4),  # Subtract 1.0 added in script
                "title": title,
                "content_snippet": content[:300] + '...' if len(content) > 300 else content,
                "metadata": {
                    "type": "cloud_provider",
                    "provider_name": hit['_source'].get('provider_name'),
                    "advantages": hit['_source'].get('advantages', []),
                    "weaknesses": hit['_source'].get('weaknesses', []),
                    "core_features": hit['_source'].get('core_features', []),
                    "use_cases": hit['_source'].get('use_cases', [])
                }
            }
            logger.info(f"Cloud result {i}: processed title='{result['title']}', content_length={len(result['content_snippet'])}, similarity={result['similarity_score']}")
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
        
        # Process innovative ideas results
        for hit in ideas_response['hits']['hits']:
            result = {
                "document_id": hit['_id'],
                "index": INNOVATIVE_IDEAS_INDEX,
                "similarity_score": round(hit['_score'] - 1.0, 4),  # Subtract 1.0 added in script
                "title": hit['_source'].get('title', 'Unknown Innovation'),
                "content_snippet": hit['_source'].get('summary', '')[:300] + '...' if len(hit['_source'].get('summary', '')) > 300 else hit['_source'].get('summary', ''),
                "metadata": {
                    "type": hit['_source'].get('type', 'innovation'),
                    "category": hit['_source'].get('category'),
                    "impact_level": hit['_source'].get('impact_level'),
                    "implementation_effort": hit['_source'].get('implementation_effort'),
                    "examples": hit['_source'].get('examples', []),
                    "tags": hit['_source'].get('tags', []),
                    "business_applications": hit['_source'].get('business_applications', []),
                    "technology_stack": hit['_source'].get('technology_stack', []),
                    "detailed_description": hit['_source'].get('detailed_description', '')
                }
            }
            all_results.append(result)
        
        # Process tech trends results
        for hit in trends_response['hits']['hits']:
            result = {
                "document_id": hit['_id'],
                "index": TECH_TRENDS_INDEX,
                "similarity_score": round(hit['_score'] - 1.0, 4),  # Subtract 1.0 added in script
                "title": hit['_source'].get('title', 'Unknown Tech Trend'),
                "content_snippet": hit['_source'].get('summary', '')[:300] + '...' if len(hit['_source'].get('summary', '')) > 300 else hit['_source'].get('summary', ''),
                "metadata": {
                    "type": "tech_trend",
                    "category": hit['_source'].get('category'),
                    "trend_status": hit['_source'].get('trend_status'),
                    "impact_level": hit['_source'].get('impact_level'),
                    "adoption_timeline": hit['_source'].get('adoption_timeline'),
                    "related_technologies": hit['_source'].get('related_technologies', []),
                    "market_implications": hit['_source'].get('market_implications', []),
                    "business_applications": hit['_source'].get('business_applications', []),
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
            "total_documents_searched": clouds_response['hits']['total']['value'] + practices_response['hits']['total']['value'] + ideas_response['hits']['total']['value'] + trends_response['hits']['total']['value'],
            "clouds_hits": clouds_response['hits']['total']['value'],
            "practices_hits": practices_response['hits']['total']['value'],
            "ideas_hits": ideas_response['hits']['total']['value'],
            "trends_hits": trends_response['hits']['total']['value'],
            "search_algorithm": "cosine_similarity",
            "indices_searched": [CLOUDS_INDEX, BEST_PRACTICES_INDEX, INNOVATIVE_IDEAS_INDEX, TECH_TRENDS_INDEX],
            "query_timestamp": datetime.now().isoformat()
        }
        
        final_results = final_results
        total_found = len(final_results)
        
    except Exception as e:
        # Fallback to basic search if vector search fails
        logger.error(f"Vector search error: {e}")
        logger.error(f"Exception type: {type(e)}")
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
            ideas_response = es.search(index=INNOVATIVE_IDEAS_INDEX, **text_query)
            trends_response = es.search(index=TECH_TRENDS_INDEX, **text_query)
            
            final_results = []
            for hit in clouds_response['hits']['hits'] + practices_response['hits']['hits'] + ideas_response['hits']['hits'] + trends_response['hits']['hits']:
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
    
    # Debug logging
    logger.info(f"Cohesive answer generator called with:")
    logger.info(f"  - Original query: {request.original_query}")
    logger.info(f"  - Vector results count: {len(request.vector_results)}")
    logger.info(f"  - SQL context count: {len(request.sql_context) if request.sql_context else 0}")
    
    # Debug: Print the structure of the first vector result to understand the format
    if request.vector_results:
        first_result = request.vector_results[0]
        logger.info(f"  - First vector result structure: {list(first_result.keys())}")
        logger.info(f"  - First vector result sample: {str(first_result)[:200]}...")
    
    # If no vector results provided, provide helpful guidance
    if not request.vector_results:
        logger.warning("No vector results provided to cohesive answer generator")
        return CohesiveAnswerResponse(
            answer=f"""To generate a comprehensive answer for "{request.original_query}", I need vector search results from the knowledge base.

It appears no vector search results were provided. To get a complete answer:

1. **Use the combined research pipeline** (`/research-pipeline`) which automatically handles all steps
2. **Or perform vector search first** (`/vector-search`) then pass those results here
3. **Or try a simpler query** that might match available data better

The cohesive answer generator works best when it has semantic search results to synthesize.""",
            sources=[],
            confidence_score=0.1,
            additional_recommendations=[
                "Use the /research-pipeline endpoint for automated processing",
                "Perform vector search first, then use those results",
                "Ensure your query matches available data in the knowledge base"
            ]
        )
    
    # Data quality validation
    valid_vector_results = []
    sources = []
    
    # Filter and validate vector results
    for i, result in enumerate(request.vector_results):
        # Check for minimum data quality
        title = result.get("title", "").strip()
        content = result.get("content_snippet", "").strip()
        similarity_score = result.get("similarity_score", 0)
        
        logger.info(f"Vector result {i}: title='{title}', content_length={len(content)}, similarity={similarity_score}")
        
        # Only include results with meaningful content and reasonable similarity
        if title and title != "Unknown Title" and content and float(similarity_score) > 0.1:
            valid_vector_results.append(result)
            
            metadata = result.get("metadata", {})
            source = {
                "title": title,
                "type": metadata.get("type", "search_result"),
                "similarity_score": str(similarity_score),
                "url": f"#{result.get('document_id', 'unknown')}"
            }
            sources.append(source)
            logger.info(f"Vector result {i}: VALID - included in analysis")
        else:
            logger.warning(f"Vector result {i}: INVALID - title_empty={not title}, unknown_title={title=='Unknown Title'}, no_content={not content}, low_similarity={float(similarity_score) <= 0.1}")
    
    logger.info(f"Data validation complete: {len(valid_vector_results)} valid results out of {len(request.vector_results)} total")
    
    # Check if we have sufficient data for analysis
    has_valid_vector_data = len(valid_vector_results) > 0
    has_sql_data = request.sql_context and len(request.sql_context) > 0
    
    if not has_valid_vector_data and not has_sql_data:
        # No valid data case - provide helpful fallback
        return CohesiveAnswerResponse(
            answer=f"""I understand you're asking about: "{request.original_query}"

Unfortunately, the current search didn't return sufficient relevant data from our knowledge base to provide a comprehensive answer. This could be due to:

1. **Limited data coverage** - The specific topic may not be well-covered in our current database
2. **Query specificity** - The question might be too specific or use terminology not present in our sources
3. **Technical issues** - There may be temporary connectivity issues with our search systems

**General guidance for your question:**
Based on the nature of your query, I recommend:
- Searching for broader, related terms first to understand the domain
- Consulting official documentation for the technologies or services mentioned
- Reaching out to community forums or expert networks for specialized advice""",
            sources=[],
            confidence_score=0.1,
            additional_recommendations=[
                "Try rephrasing your question with broader or more common terms",
                "Break down complex questions into smaller, more specific parts",
                "Consult official documentation for the technologies mentioned",
                "Consider reaching out to domain experts or community forums"
            ]
        )
    
    try:
        # Prepare high-quality context for OpenAI
        results_context = ""
        
        # Add valid vector search results
        if valid_vector_results:
            results_context += f"KNOWLEDGE BASE RESULTS ({len(valid_vector_results)} relevant sources):\n\n"
            for i, result in enumerate(valid_vector_results[:8]):  # Limit to top 8 for context
                similarity_score = float(result.get('similarity_score', 0))
                title = result.get('title', 'Unknown')
                content = result.get('content_snippet', '')
                result_type = result.get('metadata', {}).get('type', 'unknown')
                
                results_context += f"""
Source {i+1} (Relevance: {similarity_score:.3f}):
Title: {title}
Type: {result_type}
Content: {content}

"""
        
        # Add SQL context if available
        if has_sql_data:
            results_context += f"\nSTRUCTURED DATA INSIGHTS ({len(request.sql_context)} records):\n\n"
            
            for i, sql_result in enumerate(request.sql_context[:5]):  # Limit to 5 for context
                results_context += f"Record {i+1}:\n"
                for key, value in sql_result.items():
                    if key != 'id' and value is not None:  # Skip internal ID fields and null values
                        clean_key = str(key).replace('_', ' ').title()
                        results_context += f"  {clean_key}: {value}\n"
                results_context += "\n"
        
        # Generate comprehensive answer using OpenAI
        if openai_client:
            synthesis_prompt = f"""You are an expert research analyst. Provide a comprehensive answer based on the available data.

User Question: {request.original_query}

Available Research Data:
{results_context}

Instructions:
1. **Direct Answer**: Address the user's question directly using the available data
2. **Synthesize Insights**: Combine information from multiple sources when relevant
3. **Be Specific**: Use concrete details, metrics, and examples from the data
4. **Acknowledge Limitations**: Note if data is limited or if certain aspects aren't covered
5. **Actionable Advice**: Provide practical recommendations based on the findings

Format your response as:
ANSWER:
[comprehensive answer here]

RECOMMENDATIONS:
- [actionable recommendation 1]
- [actionable recommendation 2]
- [actionable recommendation 3]
"""
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research synthesis expert who creates comprehensive, actionable answers from search results."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Parse the response
            if "RECOMMENDATIONS:" in full_response:
                answer_part, recs_part = full_response.split("RECOMMENDATIONS:", 1)
                answer = answer_part.replace("ANSWER:", "").strip()
                recommendations = [
                    line.strip().lstrip("- ") 
                    for line in recs_part.split('\n') 
                    if line.strip() and line.strip().startswith('-')
                ]
            else:
                answer = full_response.replace("ANSWER:", "").strip()
                recommendations = [
                    "Review the source materials for additional details",
                    "Consider your specific requirements when applying these insights"
                ]
        else:
            # Fallback when OpenAI is not available
            data_summary = []
            if valid_vector_results:
                data_summary.append(f"{len(valid_vector_results)} relevant knowledge base sources")
            if has_sql_data:
                data_summary.append(f"{len(request.sql_context)} structured data records")
            
            answer = f"""Based on your question about "{request.original_query}", I found {' and '.join(data_summary)}.

The available data provides insights into your question, though a complete AI-powered synthesis is currently unavailable. The sources include information that addresses various aspects of your query.

Key findings from the data:
- Multiple relevant sources were identified with good similarity scores
- The information spans different categories and perspectives
- Both qualitative insights and quantitative data are available

For the most comprehensive understanding, I recommend reviewing the individual sources provided."""
            
            recommendations = [
                "Review each source individually for detailed insights",
                "Cross-reference information from multiple sources",
                "Consider how the findings apply to your specific context"
            ]
        
        # Calculate confidence score
        if valid_vector_results:
            avg_similarity = sum(float(r.get('similarity_score', 0)) for r in valid_vector_results) / len(valid_vector_results)
            result_quality = min(len(valid_vector_results) / 5, 1.0)  # Normalize based on 5 being good
            confidence_score = min((avg_similarity * 0.8 + result_quality * 0.2), 0.9)
        else:
            confidence_score = 0.6 if has_sql_data else 0.2
        
    except Exception as e:
        logger.error(f"Error in cohesive answer generator: {str(e)}")
        
        # Final fallback
        answer = f"""I encountered an issue while processing the research data for your question: "{request.original_query}"

Available data summary:
- Vector search results: {len(request.vector_results)} sources found
- Structured data: {len(request.sql_context)} records available

While I cannot provide the full AI-powered analysis at this time, the search did identify relevant sources that may help answer your question. I recommend reviewing the sources directly for insights."""
        
        recommendations = [
            "Review the individual search results for relevant information",
            "Try rephrasing your question for better search results",
            "Consider breaking complex questions into smaller parts"
        ]
        confidence_score = 0.3
    
    return CohesiveAnswerResponse(
        answer=answer,
        sources=sources,
        confidence_score=confidence_score,
        additional_recommendations=recommendations
    )

@app.post("/cohesive-answer-auto",
          response_model=CohesiveAnswerResponse,
          summary="Auto Cohesive Answer Generator", 
          description="Automatically performs vector search then generates comprehensive answer")
async def cohesive_answer_auto(request: CohesiveAnswerAutoRequest) -> CohesiveAnswerResponse:
    """
    Convenience endpoint that automatically:
    1. Performs vector search on the query
    2. Generates cohesive answer from the results
    
    This avoids the need to manually chain /vector-search -> /cohesive-answer
    """
    
    try:
        # Step 1: Perform vector search
        vector_request = VectorSearchRequest(
            clarified_query=request.query,
            sql_results=[],
            max_results=request.max_results
        )
        vector_response = await vector_search(vector_request)
        
        # Step 2: Generate cohesive answer
        answer_request = CohesiveAnswerRequest(
            original_query=request.query,
            vector_results=vector_response.results,
            sql_context=[]
        )
        
        return await cohesive_answer_generator(answer_request)
        
    except Exception as e:
        logger.error(f"Error in auto cohesive answer: {str(e)}")
        return CohesiveAnswerResponse(
            answer=f"Error processing query '{request.query}': {str(e)}",
            sources=[],
            confidence_score=0.0,
            additional_recommendations=["Try using the /research-pipeline endpoint for full automation"]
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
        
        # Step 2: Cloud Ratings SQL Execution (for structured data insights)
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
                cloud_sql_request = CloudRatingsSQLRequest(
                    question=clarified_query,
                    execute_query=True,
                    query_type="analysis"
                )
                cloud_sql_response = await cloud_ratings_sql_generator(cloud_sql_request)
                cloud_sql_results = cloud_sql_response.query_results or []
                cloud_sql_metadata = {
                    "sql_query": cloud_sql_response.sql_query,
                    "row_count": cloud_sql_response.row_count,
                    "execution_time": cloud_sql_response.execution_time,
                    "query_type": cloud_sql_response.query_type,
                    "confidence_score": cloud_sql_response.confidence_score,
                    "execution_success": cloud_sql_response.execution_success
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
            "cloud_ratings_sql_metadata": cloud_sql_metadata,
            "cloud_ratings_results_count": len(cloud_sql_results),
            "vector_results_found": vector_response.total_found,
            "vector_search_metadata": vector_response.search_metadata,
            "pipeline_version": "3.0.0",  # Updated version with combined SQL endpoint
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

@app.get("/diagnostic/elasticsearch", summary="Elasticsearch Diagnostic", description="Diagnose Elasticsearch connection and indices")
async def elasticsearch_diagnostic():
    """Diagnostic endpoint for Elasticsearch issues on Railway"""
    
    result = {
        "status": "checking",
        "elasticsearch": {},
        "indices": {},
        "environment": {},
        "recommendations": []
    }
    
    # Check environment variables
    result["environment"] = {
        "ES_URL": ES_URL,
        "ES_USER": ES_USER,
        "CLOUD_ID": "Set" if CLOUD_ID else "Not set",
        "API_KEY": "Set" if API_KEY else "Not set",
        "OPENAI_API_KEY": "Set" if OPENAI_API_KEY else "Not set",
        "CLOUDS_INDEX": CLOUDS_INDEX,
        "BEST_PRACTICES_INDEX": BEST_PRACTICES_INDEX
    }
    
    # Test Elasticsearch connection
    try:
        if es and es.ping():
            info = es.info()
            result["elasticsearch"] = {
                "connected": True,
                "version": info.get('version', {}).get('number', 'unknown'),
                "cluster_name": info.get('cluster_name', 'unknown')
            }
        else:
            result["elasticsearch"] = {
                "connected": False,
                "error": "Cannot ping Elasticsearch"
            }
            result["recommendations"].append("Check Elasticsearch connection settings")
            
    except Exception as e:
        result["elasticsearch"] = {
            "connected": False,
            "error": str(e)
        }
        result["recommendations"].append(f"Elasticsearch connection error: {str(e)}")
    
    # Check indices if connected
    if result["elasticsearch"].get("connected"):
        expected_indices = [
            CLOUDS_INDEX,
            BEST_PRACTICES_INDEX,
            INNOVATIVE_IDEAS_INDEX,
            TECH_TRENDS_INDEX
        ]
        
        result["indices"] = {
            "expected": expected_indices,
            "existing": [],
            "missing": [],
            "details": {}
        }
        
        for index_name in expected_indices:
            try:
                if es.indices.exists(index=index_name):
                    count_result = es.count(index=index_name)
                    doc_count = count_result.get('count', 0)
                    result["indices"]["existing"].append(index_name)
                    result["indices"]["details"][index_name] = {
                        "exists": True,
                        "document_count": doc_count
                    }
                else:
                    result["indices"]["missing"].append(index_name)
                    result["indices"]["details"][index_name] = {
                        "exists": False,
                        "document_count": 0
                    }
            except Exception as e:
                result["indices"]["missing"].append(index_name)
                result["indices"]["details"][index_name] = {
                    "exists": False,
                    "error": str(e)
                }
        
        # Add recommendations based on findings
        if result["indices"]["missing"]:
            result["recommendations"].append("Missing indices detected - need to upload data")
            result["recommendations"].append("Run upload script: python upload_data_multi_model.py")
            result["recommendations"].append("Or use the /upload-data endpoint if available")
            result["status"] = "indices_missing"
        else:
            result["status"] = "healthy"
    
    # List all existing indices
    try:
        if es and es.ping():
            all_indices = es.indices.get_alias("*")
            result["all_indices"] = list(all_indices.keys()) if all_indices else []
        else:
            result["all_indices"] = []
    except:
        result["all_indices"] = []
    
    return result

@app.get("/", summary="Service Information", description="Get information about the service")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Research Query Processing Tools",
        "version": "1.0.0",
        "description": "AI-powered tools for watsonx Orchestrate",
        "endpoints": [
            "/clarification-agent",
            "/cloud-ratings-sql", 
            "/vector-search",
            "/cohesive-answer",
            "/cohesive-answer-auto",
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

