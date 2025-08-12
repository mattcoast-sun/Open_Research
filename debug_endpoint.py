#!/usr/bin/env python3
"""
Debug script to examine the detailed response from the combined endpoint
"""

import requests
import json

def debug_endpoint():
    """Debug the combined endpoint response"""
    
    print("🔍 Debugging Combined Endpoint Response")
    print("=" * 50)
    
    payload = {
        "question": "Show me the top 3 cloud providers by aggregate score",
        "query_type": "ranking", 
        "execute_query": True
    }
    
    try:
        response = requests.post("http://localhost:8080/cloud-ratings-sql", json=payload)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n📋 Full Response Structure:")
            print("-" * 30)
            
            # Print each field with its value and type
            for key, value in data.items():
                value_type = type(value).__name__
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:97] + "..."
                else:
                    display_value = value
                    
                print(f"  {key}: {display_value} ({value_type})")
            
            print(f"\n🔍 Detailed Analysis:")
            print(f"  • SQL Query Generated: {bool(data.get('sql_query'))}")
            print(f"  • Execution Requested: {payload.get('execute_query')}")
            print(f"  • Execution Success Field: {data.get('execution_success')}")
            print(f"  • Query Results Field: {data.get('query_results')}")
            print(f"  • Row Count Field: {data.get('row_count')}")
            print(f"  • Execution Time Field: {data.get('execution_time')}")
            
            # Pretty print the JSON for inspection
            print(f"\n📝 Raw JSON Response:")
            print("-" * 30)
            print(json.dumps(data, indent=2))
            
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    debug_endpoint()
