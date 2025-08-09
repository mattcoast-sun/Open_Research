#!/usr/bin/env python3
"""
Cloud Ratings SQL Endpoint - Usage Examples

This script demonstrates how to use the /cloud-ratings-sql endpoint
to generate SQL queries for the cloud ratings database.
"""

import requests
import json


def generate_sql_query(question: str, query_type: str = "analysis", base_url: str = "http://localhost:8000"):
    """Generate a SQL query for the cloud ratings database."""
    
    endpoint = f"{base_url}/cloud-ratings-sql"
    
    payload = {
        "question": question,
        "query_type": query_type
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except Exception as e:
        return {"error": str(e)}


def main():
    """Demonstrate the endpoint with example queries."""
    
    print("Cloud Ratings SQL Query Generator - Examples")
    print("=" * 60)
    
    # Example questions
    examples = [
        {
            "question": "Which are the top 3 cloud providers overall?",
            "type": "ranking"
        },
        {
            "question": "Compare AWS and GCP on all metrics",  
            "type": "comparison"
        },
        {
            "question": "What's the average cost efficiency score?",
            "type": "analysis"
        },
        {
            "question": "Show me everything about Azure",
            "type": "specific"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example['question']}")
        print("-" * 50)
        
        result = generate_sql_query(example["question"], example["type"])
        
        if "error" not in result:
            print(f"🔍 Query Type: {result.get('query_type')}")
            print(f"📊 Generated SQL:")
            print(result.get('sql_query', ''))
            print(f"\n💡 Explanation:")
            print(result.get('explanation', ''))
        else:
            print(f"❌ Error: {result['error']}")
    
    print(f"\n" + "=" * 60)
    print("💻 HOW TO USE THIS ENDPOINT:")
    print("=" * 60)
    
    print("""
1. Start the FastAPI server:
   python main.py

2. Make a POST request to /cloud-ratings-sql:

   import requests
   
   response = requests.post('http://localhost:8000/cloud-ratings-sql', json={
       "question": "What are the top 5 cloud providers?",
       "query_type": "ranking"
   })
   
   result = response.json()
   print(result['sql_query'])

3. Use the generated SQL with your SQLite database:

   import sqlite3
   
   conn = sqlite3.connect('cloud_ratings.db')
   cursor = conn.cursor()
   
   cursor.execute(result['sql_query'])
   rows = cursor.fetchall()
   
   for row in rows:
       print(row)
   
   conn.close()

4. Available query types:
   - "ranking": For top/best/worst queries
   - "comparison": For comparing specific providers
   - "analysis": For statistical analysis and filtering
   - "specific": For information about particular providers
    """)

if __name__ == "__main__":
    main()
