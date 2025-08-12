#!/usr/bin/env python3
"""
Railway Deployment Test Suite

This test suite verifies that the Railway deployment is working correctly
after the Elasticsearch index naming fix.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

# Railway deployment URL
BASE_URL = "https://researcherlegend-production.up.railway.app"

class RailwayTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Railway-Test-Suite/1.0'
        })
        
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = "", response_data: Any = None):
        """Log a test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "response_data": response_data
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        print("\n🏥 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Health Check", True, f"Status: {data.get('status')}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unhealthy status: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_diagnostic_endpoint(self) -> bool:
        """Test the diagnostic endpoint (if deployed)"""
        print("\n🔍 Testing Diagnostic Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/diagnostic/elasticsearch", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if ES is connected
                es_connected = data.get("elasticsearch", {}).get("connected", False)
                if es_connected:
                    indices = data.get("indices", {})
                    missing = indices.get("missing", [])
                    existing = indices.get("existing", [])
                    
                    if not missing:
                        self.log_test("Diagnostic - All Indices Present", True, 
                                    f"Found: {existing}")
                        return True
                    else:
                        self.log_test("Diagnostic - Missing Indices", False, 
                                    f"Missing: {missing}")
                        return False
                else:
                    self.log_test("Diagnostic - Elasticsearch", False, "ES not connected")
                    return False
                    
            elif response.status_code == 404:
                self.log_test("Diagnostic Endpoint", False, "Endpoint not deployed yet")
                return False
            else:
                self.log_test("Diagnostic Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Diagnostic Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_vector_search_endpoint(self) -> bool:
        """Test the vector search endpoint"""
        print("\n🔍 Testing Vector Search Endpoint...")
        
        test_payload = {
            "clarified_query": "cloud security best practices",
            "max_results": 5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/vector-search", 
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                total_found = data.get("total_found", 0)
                
                if results and total_found > 0:
                    self.log_test("Vector Search", True, 
                                f"Found {total_found} results, returned {len(results)}")
                    return True
                else:
                    self.log_test("Vector Search", False, "No results returned")
                    return False
            else:
                error_text = response.text[:200] if response.text else "No error details"
                self.log_test("Vector Search", False, 
                            f"HTTP {response.status_code}: {error_text}")
                return False
                
        except Exception as e:
            self.log_test("Vector Search", False, f"Exception: {str(e)}")
            return False
    
    def test_research_pipeline_endpoint(self) -> bool:
        """Test the main research pipeline endpoint"""
        print("\n🔬 Testing Research Pipeline Endpoint...")
        
        test_payload = {
            "question": "What are the best cloud providers for AI workloads?",
            "skip_clarification": True,
            "max_results": 5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/research-pipeline", 
                json=test_payload,
                timeout=45  # Longer timeout for full pipeline
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["original_question", "final_answer", "sources", "confidence_score"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    answer_length = len(data.get("final_answer", ""))
                    sources_count = len(data.get("sources", []))
                    confidence = data.get("confidence_score", 0)
                    
                    self.log_test("Research Pipeline", True, 
                                f"Answer: {answer_length} chars, Sources: {sources_count}, Confidence: {confidence:.2f}")
                    return True
                else:
                    self.log_test("Research Pipeline", False, 
                                f"Missing fields: {missing_fields}")
                    return False
            else:
                error_text = response.text[:300] if response.text else "No error details"
                self.log_test("Research Pipeline", False, 
                            f"HTTP {response.status_code}: {error_text}")
                return False
                
        except Exception as e:
            self.log_test("Research Pipeline", False, f"Exception: {str(e)}")
            return False
    
    def test_cloud_ratings_sql_endpoint(self) -> bool:
        """Test the cloud ratings SQL endpoint"""
        print("\n📊 Testing Cloud Ratings SQL Endpoint...")
        
        test_payload = {
            "question": "Show me the top 3 cloud providers by aggregate score",
            "execute_query": True,
            "query_type": "ranking"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/cloud-ratings-sql", 
                json=test_payload,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                
                sql_query = data.get("sql_query", "")
                query_results = data.get("query_results", [])
                execution_success = data.get("execution_success")
                
                if sql_query and execution_success:
                    result_count = len(query_results) if query_results else 0
                    self.log_test("Cloud Ratings SQL", True, 
                                f"Generated query, executed successfully, {result_count} results")
                    return True
                elif sql_query and execution_success is False:
                    self.log_test("Cloud Ratings SQL", False, 
                                "Query generated but execution failed")
                    return False
                else:
                    self.log_test("Cloud Ratings SQL", False, 
                                "No SQL query generated")
                    return False
            else:
                error_text = response.text[:200] if response.text else "No error details"
                self.log_test("Cloud Ratings SQL", False, 
                            f"HTTP {response.status_code}: {error_text}")
                return False
                
        except Exception as e:
            self.log_test("Cloud Ratings SQL", False, f"Exception: {str(e)}")
            return False
    
    def test_cohesive_answer_auto(self) -> bool:
        """Test the auto cohesive answer endpoint"""
        print("\n🤖 Testing Auto Cohesive Answer Endpoint...")
        
        test_payload = {
            "query": "What are the advantages of serverless computing?",
            "max_results": 5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/cohesive-answer-auto", 
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                confidence = data.get("confidence_score", 0)
                
                if answer and len(answer) > 50:  # Meaningful answer
                    self.log_test("Auto Cohesive Answer", True, 
                                f"Answer: {len(answer)} chars, Sources: {len(sources)}, Confidence: {confidence:.2f}")
                    return True
                else:
                    self.log_test("Auto Cohesive Answer", False, 
                                f"Short/empty answer: {len(answer)} chars")
                    return False
            else:
                error_text = response.text[:200] if response.text else "No error details"
                self.log_test("Auto Cohesive Answer", False, 
                            f"HTTP {response.status_code}: {error_text}")
                return False
                
        except Exception as e:
            self.log_test("Auto Cohesive Answer", False, f"Exception: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test response time performance"""
        print("\n⚡ Testing Performance...")
        
        test_cases = [
            ("Health Check", "GET", "/health", None),
            ("Vector Search", "POST", "/vector-search", {"clarified_query": "test query", "max_results": 3})
        ]
        
        all_good = True
        
        for test_name, method, endpoint, payload in test_cases:
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=15)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200 and response_time < 10.0:  # 10 second threshold
                    self.log_test(f"Performance - {test_name}", True, 
                                f"{response_time:.2f}s response time")
                else:
                    self.log_test(f"Performance - {test_name}", False, 
                                f"{response_time:.2f}s (too slow or failed)")
                    all_good = False
                    
            except Exception as e:
                self.log_test(f"Performance - {test_name}", False, f"Exception: {str(e)}")
                all_good = False
        
        return all_good
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🧪 Railway Deployment Test Suite")
        print("=" * 60)
        print(f"Testing: {self.base_url}")
        print()
        
        # Run all tests
        tests = [
            self.test_health_endpoint,
            self.test_diagnostic_endpoint,
            self.test_vector_search_endpoint,
            self.test_cloud_ratings_sql_endpoint,
            self.test_cohesive_answer_auto,
            self.test_research_pipeline_endpoint,
            self.test_performance
        ]
        
        for test_func in tests:
            try:
                test_func()
                time.sleep(1)  # Brief pause between tests
            except KeyboardInterrupt:
                print("\n⏹️ Tests interrupted by user")
                break
            except Exception as e:
                print(f"\n💥 Unexpected error in {test_func.__name__}: {e}")
        
        # Print summary
        self.print_summary()
        
        return self.failed == 0
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! Railway deployment is working correctly.")
        else:
            print(f"\n⚠️ {self.failed} test(s) failed. Check the details above.")
        
        print("\n📋 Individual Test Results:")
        for result in self.test_results:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test']}")
            if result["details"]:
                print(f"      {result['details']}")

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Railway deployment")
    parser.add_argument("--url", default=BASE_URL, help="Base URL to test")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests")
    
    args = parser.parse_args()
    
    tester = RailwayTester(args.url)
    
    if args.quick:
        print("🚀 Running Quick Test Suite...")
        success = (
            tester.test_health_endpoint() and
            tester.test_vector_search_endpoint() and
            tester.test_research_pipeline_endpoint()
        )
        tester.print_summary()
    else:
        success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
