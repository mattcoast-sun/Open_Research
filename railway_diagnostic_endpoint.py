"""
Railway Diagnostic Endpoint

Add this to your main.py to diagnose Elasticsearch issues on Railway
"""

from fastapi import FastAPI
from typing import Dict, Any
import os
from elasticsearch import Elasticsearch

# Add this endpoint to your existing FastAPI app
def add_diagnostic_endpoint(app: FastAPI, es: Elasticsearch):
    """Add diagnostic endpoint to existing FastAPI app"""
    
    @app.get("/diagnostic/elasticsearch", 
             summary="Elasticsearch Diagnostic", 
             description="Diagnose Elasticsearch connection and indices on Railway")
    async def elasticsearch_diagnostic() -> Dict[str, Any]:
        """Diagnostic endpoint for Elasticsearch issues"""
        
        result = {
            "status": "checking",
            "elasticsearch": {},
            "indices": {},
            "environment": {},
            "recommendations": []
        }
        
        # Check environment variables
        result["environment"] = {
            "ES_URL": os.getenv("ES_URL", "Not set"),
            "ES_USER": os.getenv("ES_USER", "Not set"), 
            "CLOUD_ID": "Set" if os.getenv("CLOUD_ID") else "Not set",
            "API_KEY": "Set" if os.getenv("ES_API_KEY") else "Not set",
            "OPENAI_API_KEY": "Set" if os.getenv("OPENAI_API_KEY") else "Not set"
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
                "cloud_providers_providers",
                "cloud_providers_best_practices",
                "innovative_ideas", 
                "tech_trends_innovation"
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
                result["recommendations"].append("Run upload script to create missing indices")
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

# Simple standalone diagnostic for Railway
async def simple_diagnostic():
    """Simple diagnostic that can be run independently"""
    
    # Import Elasticsearch setup from main
    try:
        import main
        es = main.es  # Use the es client from main.py
    except:
        # Fallback - create our own client
        from elasticsearch import Elasticsearch
        
        ES_URL = os.getenv("ES_URL", "http://localhost:9200")
        ES_USER = os.getenv("ES_USER", "elastic")
        ES_PASS = os.getenv("ES_PASS", "password")
        CLOUD_ID = os.getenv("CLOUD_ID")
        API_KEY = os.getenv("ES_API_KEY")
        
        try:
            if CLOUD_ID and API_KEY:
                es = Elasticsearch(cloud_id=CLOUD_ID, api_key=API_KEY)
            else:
                es = Elasticsearch(ES_URL, http_auth=(ES_USER, ES_PASS), verify_certs=False)
        except Exception as e:
            return {"error": f"Cannot create Elasticsearch client: {e}"}
    
    # Run diagnostic
    result = {}
    
    # Test connection
    try:
        if es.ping():
            result["elasticsearch_connected"] = True
            
            # Check indices
            expected_indices = [
                "cloud_providers_providers",
                "cloud_providers_best_practices", 
                "innovative_ideas",
                "tech_trends_innovation"
            ]
            
            missing_indices = []
            existing_indices = []
            
            for index in expected_indices:
                if es.indices.exists(index=index):
                    existing_indices.append(index)
                else:
                    missing_indices.append(index)
            
            result["existing_indices"] = existing_indices
            result["missing_indices"] = missing_indices
            
            if missing_indices:
                result["status"] = "MISSING_INDICES"
                result["action_needed"] = "Upload data to create missing indices"
            else:
                result["status"] = "HEALTHY"
                
        else:
            result["elasticsearch_connected"] = False
            result["status"] = "CONNECTION_FAILED"
            
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "ERROR"
    
    return result
