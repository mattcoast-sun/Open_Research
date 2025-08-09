#!/usr/bin/env python3
"""
Simple script to run the FastAPI server
Make sure to activate the virtual environment first:
source venv/bin/activate
"""
"""
Simple script to run the FastAPI server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "ES_URL", "ES_USER", "ES_PASS"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file.")
        exit(1)
    
    print("🚀 Starting FastAPI server...")
    print("📊 Dashboard: http://localhost:8001/docs")
    print("🔍 Health check: http://localhost:8001/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
