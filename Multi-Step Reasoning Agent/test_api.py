#!/usr/bin/env python3
"""
Test script for the API endpoint.
Run this after starting the API server.
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_solve(question, mock=False):
    """Test solve endpoint"""
    print(f"Testing /solve endpoint with question: {question}")
    print(f"Mock mode: {mock}\n")
    
    payload = {
        "question": question,
        "mock": mock
    }
    
    response = requests.post(
        f"{API_URL}/solve",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()

if __name__ == "__main__":
    import sys
    
    # Test health
    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running:")
        print("   python3 api.py")
        sys.exit(1)
    
    # Test solve with mock
    print("=" * 60)
    test_solve("If a train leaves at 14:30 and arrives at 18:05, how long is the journey?", mock=True)
    
    # Test solve with real API (if key is set)
    if not sys.argv or "--mock-only" not in sys.argv:
        print("=" * 60)
        test_solve("Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?", mock=False)

