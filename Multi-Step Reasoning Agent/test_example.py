#!/usr/bin/env python3
"""
Quick test script to verify the agent works.
Run with --mock for testing without API key, or set GOOGLE_API_KEY for real testing.
"""

import sys
import os
from interface import solve

def main():
    # Check if mock mode
    mock = "--mock" in sys.argv or os.getenv("GOOGLE_API_KEY") is None
    
    if mock:
        print("Running in MOCK mode (no API key needed)")
    else:
        print("Running with REAL Gemini API")
    
    # Test question
    question = "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"
    
    print(f"\nQuestion: {question}\n")
    print("Processing...\n")
    
    result = solve(question, mock=mock)
    
    print("Result:")
    import json
    print(json.dumps(result, indent=2))
    
    if result["status"] == "success":
        print(f"\n✓ Success! Answer: {result['answer']}")
    else:
        print(f"\n✗ Failed: {result['reasoning_visible_to_user']}")

if __name__ == "__main__":
    main()

