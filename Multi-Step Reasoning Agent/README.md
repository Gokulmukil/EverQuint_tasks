# Multi-Step Reasoning Agent

A Python-based agent for solving word problems with planning, execution, and verification phases. Uses Google Gemini Flash 2.5 (mock mode available). Outputs JSON as specified.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set `GOOGLE_API_KEY` env var for real LLM (optional; mock works).
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Run CLI: `python3 interface.py --question "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?" [--mock]`
4. Run in notebook: Jupyter with `from interface import solve; solve("question")`
5. Tests: `python3 run_tests.py` (generates examples/run_logs.jsonl)

## Model Configuration
The agent uses Gemini Flash 2.5 (model: `gemini-2.5-flash`) by default. To use a different model, modify the `model` parameter in `llm_interface.py` or pass it when creating `LLMInterface`.

## Structure
- `prompts.py`: Templated prompts.
- `llm_interface.py`: LLM caller (Gemini Flash 2.5 or mock).
- `planner.py`/`executor.py`/`verifier.py`: Core phases.
- `agent.py`: Orchestrator with retry loop.
- `interface.py`: CLI + notebook entrypoint.
- `tests/test_agent.py`: Pytest suite (run via run_tests.py).

## Prompt Rationale
- **Planner**: JSON steps for parsability; concise to avoid bloat.
- **Executor**: Step-bound to reduce drift; intermediates for verification.
- **Verifier**: Multi-check (LLM re-solve + code constraints); issues list for debug.
Tried: Free-text CoT (parsing errors); improved: JSON-only with low temp.
With time: Add few-shots, adaptive plans.

## Assumptions
- Times in 24h format; basic arithmetic only.
- Verifier stubs for domains (extendable).
- Logs to agent.log; outputs to stdout/JSON.

## Examples
See examples/run_logs.jsonl for sample runs.

## API Server

The agent can also be run as a REST API server:

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or use virtual environment:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set API key (optional, for real API):
   ```bash
   export GOOGLE_API_KEY=your_key_here
   ```

3. Start the server:
   ```bash
   python3 api.py
   # Or use the helper script:
   ./start_api.sh
   ```
   The server will run on `http://localhost:5000`

### API Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /solve` - Solve a word problem

### Example Requests

**Using curl:**
```bash
curl -X POST http://localhost:5000/solve \
  -H "Content-Type: application/json" \
  -d '{"question": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?", "mock": false}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:5000/solve",
    json={
        "question": "Your question here",
        "mock": False  # Set to True for testing without API key
    }
)
print(response.json())
```

**Request Payload:**
```json
{
  "question": "Your word problem question",
  "mock": false
}
```

**Response Format:**
```json
{
  "answer": "The answer to the question",
  "status": "success" | "failed",
  "reasoning_visible_to_user": "Explanation of the solution",
  "metadata": {
    "plan": "Step 1 | Step 2 | Step 3...",
    "checks": [...],
    "retries": 0
  }
}
```

### Testing

Run the test script (make sure server is running):
```bash
python3 test_api.py
# Or use the example script:
python3 api_example.py
```

