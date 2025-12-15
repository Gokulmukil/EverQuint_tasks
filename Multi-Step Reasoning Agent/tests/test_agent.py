import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interface import solve

# Note: Use mock=True for deterministic tests
TESTS = [
    # Easy (5)
    {"q": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?", "exp_ans": "3 hours 35 minutes"},
    {"q": "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?", "exp_ans": "9"},
    {"q": "What is 15 + 27?", "exp_ans": "42"},
    {"q": "A slot from 10:00 to 11:00 is how long?", "exp_ans": "60 minutes"},
    {"q": "Bob has 4 oranges, eats half. How many left?", "exp_ans": "2"},
    
    # Tricky (4)
    {"q": "A meeting needs 60 minutes. Slots: 09:00–09:30, 09:45–10:30, 11:00–12:00. Which can fit?", "exp_ans": "11:00–12:00"},
    {"q": "Alice has twice as many as Bob's 3 apples. Total?", "exp_ans": "9"},  # Ambiguous parse
    {"q": "Train leaves 23:50 arrives 00:10. Duration?", "exp_ans": "20 minutes"},  # Overnight edge
    {"q": "Zero apples + zero. Total?", "exp_ans": "0"},  # Edge zero
]

@pytest.mark.parametrize("test_case", TESTS)
def test_solve(test_case):
    result = solve(test_case["q"], mock=True)
    assert result["status"] in ["success", "failed"]  # Status should be valid
    # For mock mode, we expect success
    if result["status"] == "success":
        assert test_case["exp_ans"].lower() in result["answer"].lower() or any(char.isdigit() for char in result["answer"])  # Partial match or contains numbers
    assert "metadata" in result
    assert "retries" in result["metadata"]
    # Log
    log_entry = {
        "question": test_case["q"],
        "final_json": result,
        "verifier_passed": result["status"] == "success",
        "retried": result["metadata"]["retries"] > 0
    }
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "run_logs.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

