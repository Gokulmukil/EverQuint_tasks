#!/usr/bin/env python3
"""
Example of how to use the API endpoint.
Make sure the API server is running: python3 api.py
"""

import requests
import json

API_URL = "http://localhost:5000"

# Example 1: Simple question with mock mode
print("Example 1: Testing with mock mode")
response = requests.post(
    f"{API_URL}/solve",
    json={
        "question": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
        "mock": True
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

# Example 2: Real API (requires GOOGLE_API_KEY to be set)
print("Example 2: Testing with real API")
response = requests.post(
    f"{API_URL}/solve",
    json={
        "question": "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?",
        "mock": False
    }
)
print(f"Status Code: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

# Example 3: Health check
print("Example 3: Health check")
response = requests.get(f"{API_URL}/health")
print(f"Status Code: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

