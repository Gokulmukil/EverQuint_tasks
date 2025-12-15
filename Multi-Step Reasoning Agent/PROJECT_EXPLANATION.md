# Multi-Step Reasoning Agent - Complete Code Explanation

## Project Overview

This is a **Multi-Step Reasoning Agent** that solves word problems using a three-phase approach:
1. **Planning**: Breaks down the problem into steps
2. **Execution**: Performs calculations following the plan
3. **Verification**: Validates the solution before returning it

The agent uses Google's Gemini Flash 2.5 LLM (or a mock mode for testing) and includes retry logic for robustness.

---

## Architecture Overview

```
User Question
    ↓
interface.py (Entry Point)
    ↓
agent.py (Orchestrator)
    ↓
┌─────────────┬──────────────┬──────────────┐
│ planner.py  │ executor.py  │ verifier.py  │
└─────────────┴──────────────┴──────────────┘
         ↓              ↓              ↓
    llm_interface.py (LLM Communication)
         ↓
    prompts.py (Prompt Templates)
```

---

## File-by-File Detailed Explanation

### 1. `prompts.py` - Prompt Templates

**Purpose**: Contains all the prompt templates used to communicate with the LLM.

```python
PLANNER_PROMPT = """
You are a precise planner for solving word problems. Given a question, output a JSON with a numbered list of 3-6 concise steps to solve it. Steps should cover parsing, computation, validation, and formatting. 
Example: {{"steps": ["1. Parse quantities: extract numbers and relations.", "2. Compute totals: add reds + greens.", "3. Validate: ensure non-negative.", "4. Format answer."]}}

Question: {question}
Output only JSON.
"""
```

**Explanation**:
- Uses double braces `{{}}` because `.format()` treats single braces as placeholders
- Instructs the LLM to create a JSON with a "steps" array
- Asks for 3-6 steps covering: parsing → computation → validation → formatting
- The `{question}` placeholder gets filled with the actual question

```python
EXEC_PROMPT = """
You are an executor following a strict plan. For the question and plan, perform each step in order. Show intermediate results as JSON: {{"intermediates": {{"step1": "result1", "step2": "result2"}}, "proposed_answer": "short final answer"}}. Use arithmetic/code-like notation for calcs. Do not skip steps.

Question: {question}
Plan: {plan_steps}
Output only JSON.
"""
```

**Explanation**:
- Instructs the LLM to follow the plan step-by-step
- Requires JSON output with:
  - `intermediates`: Dictionary showing results of each step
  - `proposed_answer`: The final answer
- Emphasizes not skipping steps to ensure thoroughness

```python
VERIFIER_PROMPT = """
You are a verifier checking a proposed solution for a question. Independently re-solve briefly, then check: (1) Matches proposed? (2) Constraints valid (e.g., positive nums, time logic)? (3) Consistent explanation? Output JSON: {{"approved": true/false, "issues": ["list of problems"], "re_solution": "brief alt solve"}}.

Question: {question}
Proposed: {proposed_json}
Output only JSON.
"""
```

**Explanation**:
- Asks the LLM to independently re-solve the problem
- Checks three things:
  1. Does the re-solution match the proposed answer?
  2. Are constraints valid (e.g., positive numbers, logical time calculations)?
  3. Is the explanation consistent?
- Returns JSON with approval status, issues list, and alternative solution

---

### 2. `llm_interface.py` - LLM Communication Layer

**Purpose**: Abstracts communication with Google Gemini API, with a mock mode for testing.

```python
class LLMInterface:
    def __init__(self, model="gemini-2.5-flash", mock=False):
        self.model_name = model
        self.mock = mock
        if not mock:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        else:
            self.client = None
```

**Explanation**:
- `model`: Defaults to "gemini-2.5-flash" (Google's fast model)
- `mock`: If True, skips API setup and uses hardcoded responses
- In real mode: Reads `GOOGLE_API_KEY` from environment, configures the Gemini client
- In mock mode: Sets `client` to None

```python
        self.mock_responses = {
            "planner": '{"steps": ["1. Parse the question.", "2. Extract key numbers.", "3. Perform calculation.", "4. Validate result.", "5. Format answer."]}',
            "executor": '{"intermediates": {"step1": "parsed: times 14:30 to 18:05", "step2": "diff: 3h35m", "step3": "valid", "step4": "ok"}, "proposed_answer": "3 hours 35 minutes"}',
            "verifier": '{"approved": true, "issues": [], "re_solution": "14:30 to 18:05 is 3h35m"}'
        }
```

**Explanation**:
- Pre-defined JSON responses for each phase when in mock mode
- Allows testing without API calls or costs
- Can be extended with more sophisticated mock responses

```python
    def call(self, prompt: str, system: str = "You are helpful.") -> str:
        if self.mock:
            # Basic mock: return based on keyword
            if "planner" in prompt.lower() or "precise planner" in prompt.lower():
                return self.mock_responses["planner"]
            elif "executor" in prompt.lower() or "follow" in prompt.lower():
                return self.mock_responses["executor"]
            elif "verifier" in prompt.lower():
                return self.mock_responses["verifier"]
            return '{"error": "mock fail"}'
```

**Explanation**:
- In mock mode: Checks prompt content to determine which mock response to return
- Uses keyword matching (case-insensitive) to identify the phase
- Falls back to error JSON if no match

```python
        try:
            # Combine system and user prompt for Gemini
            full_prompt = f"{system}\n\n{prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for deterministic output
                    top_p=0.95,
                    top_k=40,
                )
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")
```

**Explanation**:
- Combines system message and user prompt
- Uses low temperature (0.1) for more deterministic, consistent outputs
- `top_p=0.95` and `top_k=40` control sampling diversity
- Returns the text response from Gemini
- Wraps exceptions in a clear error message

---

### 3. `planner.py` - Planning Phase

**Purpose**: Takes a question and generates a step-by-step plan.

```python
def plan(llm: LLMInterface, question: str) -> List[str]:
    prompt = PLANNER_PROMPT.format(question=question)
    response = llm.call(prompt, system="You are a precise planner for solving word problems.")
```

**Explanation**:
- Formats the planner prompt with the actual question
- Calls the LLM with a system message emphasizing precision
- Returns the raw LLM response

```python
    try:
        # Try to extract JSON from response (in case there's extra text)
        response_clean = response.strip()
        # Remove markdown code blocks if present
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()
```

**Explanation**:
- LLMs sometimes wrap JSON in markdown code blocks (```json ... ```)
- This code strips those markers to get pure JSON
- Handles both ````json` and plain ```` markers

```python
        data = json.loads(response_clean)
        return data["steps"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid plan JSON: {e}. Response was: {response}")
```

**Explanation**:
- Parses the cleaned response as JSON
- Extracts the "steps" array
- If parsing fails or "steps" is missing, raises a clear error with the original response for debugging

**Example Output**:
```python
["1. Parse the question to extract departure and arrival times.",
 "2. Convert times to minutes for easier calculation.",
 "3. Calculate the difference: arrival - departure.",
 "4. Convert back to hours and minutes format.",
 "5. Format the final answer."]
```

---

### 4. `executor.py` - Execution Phase

**Purpose**: Follows the plan to compute the answer.

```python
def execute(llm: LLMInterface, question: str, plan_steps: List[str]) -> Dict[str, Any]:
    prompt = EXEC_PROMPT.format(question=question, plan_steps="\n".join(plan_steps))
    response = llm.call(prompt, system="You are an executor following a strict plan.")
```

**Explanation**:
- Formats the executor prompt with the question and plan steps (joined by newlines)
- System message emphasizes following the plan strictly

```python
    try:
        # Try to extract JSON from response (in case there's extra text)
        response_clean = response.strip()
        # Remove markdown code blocks if present
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()
        
        data = json.loads(response_clean)
        if "proposed_answer" not in data:
            raise ValueError("Missing proposed_answer")
        return data
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Invalid execution JSON: {e}. Response was: {response}")
```

**Explanation**:
- Same markdown stripping logic as planner
- Validates that "proposed_answer" exists in the JSON
- Returns the full dictionary (includes both "intermediates" and "proposed_answer")
- Raises clear errors if validation fails

**Example Output**:
```python
{
    "intermediates": {
        "step1": "Parsed: departure=14:30, arrival=18:05",
        "step2": "Converted: 14:30 = 870 min, 18:05 = 1085 min",
        "step3": "Difference: 1085 - 870 = 215 minutes",
        "step4": "Converted: 215 min = 3 hours 35 minutes"
    },
    "proposed_answer": "3 hours 35 minutes"
}
```

---

### 5. `verifier.py` - Verification Phase

**Purpose**: Validates the proposed solution using both LLM and code-based checks.

```python
def verify(llm: LLMInterface, question: str, proposed: Dict[str, Any]) -> Dict[str, Any]:
    # LLM check
    prompt = VERIFIER_PROMPT.format(question=question, proposed_json=json.dumps(proposed))
    try:
        response = llm.call(prompt, system="You are a verifier checking a proposed solution.")
        # ... (markdown stripping code, same as before)
        llm_check = json.loads(response_clean)
    except json.JSONDecodeError as e:
        llm_check = {"approved": False, "issues": [f"JSON parse error: {e}"], "re_solution": ""}
```

**Explanation**:
- Formats verifier prompt with question and proposed solution (as JSON string)
- If JSON parsing fails, defaults to "not approved" with an error message

```python
    # Code-based checks
    checks = []
    if "time" in question.lower() or "train" in question.lower() or "meeting" in question.lower() or "slot" in question.lower():
        duration = _parse_time_diff(question)
        checks.append({
            "check_name": "time_positive",
            "passed": duration > 0,
            "details": f"Duration: {duration} minutes"
        })
```

**Explanation**:
- Detects time-related questions by keywords
- Calls `_parse_time_diff()` to compute duration programmatically
- Checks that duration is positive (negative time doesn't make sense)
- Records the check result with name, pass/fail, and details

```python
    elif "apple" in question.lower() or "orange" in question.lower() or "total" in question.lower():
        try:
            # Try to extract number from proposed answer
            answer_text = str(proposed.get("proposed_answer", ""))
            numbers = re.findall(r'\d+', answer_text)
            if numbers:
                total = int(numbers[0])
                checks.append({
                    "check_name": "non_negative",
                    "passed": total >= 0,
                    "details": f"Total: {total}"
                })
```

**Explanation**:
- Detects counting/arithmetic questions
- Extracts numbers from the proposed answer using regex
- Checks that the number is non-negative (can't have negative apples)
- Gracefully handles cases where no number is found

```python
    # Aggregate
    llm_pass = llm_check.get("approved", False)
    code_pass = all(c["passed"] for c in checks) if checks else True
    overall_pass = llm_pass and code_pass
    issues = llm_check.get("issues", [])
    if not code_pass:
        issues.extend([c["details"] for c in checks if not c["passed"]])

    return {
        "approved": overall_pass,
        "checks": checks + [{"check_name": "llm_consistency", "passed": llm_pass, "details": str(issues)}],
        "re_solution": llm_check.get("re_solution", ""),
        "issues": issues
    }
```

**Explanation**:
- Combines LLM check and code checks: both must pass
- If code checks fail, adds their details to the issues list
- Returns comprehensive verification result with:
  - `approved`: Overall pass/fail
  - `checks`: All individual checks (code + LLM)
  - `re_solution`: LLM's alternative solution
  - `issues`: List of all problems found

```python
def _parse_time_diff(question: str) -> int:
    # Simple regex for HH:MM
    times = re.findall(r'(\d{2}:\d{2})', question)
    if len(times) >= 2:
        try:
            start = datetime.strptime(times[0], '%H:%M')
            end = datetime.strptime(times[1], '%H:%M')
            if end < start:
                end += timedelta(days=1)  # Overnight, but assume same day
            delta = end - start
            return int(delta.total_seconds() / 60)
        except:
            pass
```

**Explanation**:
- Uses regex to find all time patterns (HH:MM format)
- Parses first two times as start and end
- Handles overnight cases (e.g., 23:50 to 00:10) by adding a day
- Returns duration in minutes
- Also handles time ranges like "09:00–09:30" (with dash/em-dash)

---

### 6. `agent.py` - Main Orchestrator

**Purpose**: Coordinates the three phases and handles retries.

```python
class ReasoningAgent:
    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
```

**Explanation**:
- Stores the LLM interface and maximum retry count
- `max_retries=2` means up to 3 total attempts (initial + 2 retries)

```python
    def solve(self, question: str) -> Dict[str, Any]:
        retries = 0
        while retries <= self.max_retries:
            try:
                # Plan
                plan_steps = plan(self.llm, question)
                logging.info(f"Plan: {plan_steps}")

                # Execute
                proposed = execute(self.llm, question, plan_steps)
                logging.info(f"Proposed: {proposed}")

                # Verify
                verification = verify(self.llm, question, proposed)
                logging.info(f"Verification: {verification}")
```

**Explanation**:
- Main solve loop: runs until success or max retries
- **Phase 1**: Calls `plan()` to get steps
- **Phase 2**: Calls `execute()` with the plan
- **Phase 3**: Calls `verify()` to check the solution
- Logs each phase for debugging

```python
                if verification["approved"]:
                    return self._assemble_json(proposed["proposed_answer"], "success", plan_steps, verification["checks"], retries)
                else:
                    retries += 1
                    logging.warning(f"Retry {retries}: Issues - {verification['issues']}")
                    if retries > self.max_retries:
                        short_reasoning = f"Failed after {retries} retries: {', '.join(verification['issues'][:2])}"
                        return self._assemble_json(proposed.get("proposed_answer", "N/A"), "failed", plan_steps, verification["checks"], retries, short_reasoning)
```

**Explanation**:
- If verification passes: returns success JSON immediately
- If verification fails: increments retry counter
- If max retries exceeded: returns failure JSON with first 2 issues
- Uses `.get("proposed_answer", "N/A")` to handle missing answer gracefully

```python
            except Exception as e:
                logging.error(f"Error in iteration {retries}: {str(e)}")
                retries += 1
                if retries > self.max_retries:
                    return {
                        "answer": "Error in processing",
                        "status": "failed",
                        "reasoning_visible_to_user": f"Internal error after {retries} retries: {str(e)}",
                        "metadata": {"plan": [], "checks": [], "retries": retries}
                    }
```

**Explanation**:
- Catches any exceptions (JSON parsing errors, API failures, etc.)
- Logs the error
- If max retries exceeded, returns error JSON
- Always returns a valid response structure, never crashes

```python
    def _assemble_json(self, answer: str, status: str, plan_steps: List[str], checks: List[Dict], retries: int, reasoning: str = "") -> Dict[str, Any]:
        if not reasoning and status == "success":
            reasoning = f"Solved in {len(plan_steps)} steps with {len(checks)} checks passing."
        return {
            "answer": answer,
            "status": status,
            "reasoning_visible_to_user": reasoning,
            "metadata": {
                "plan": " | ".join(plan_steps),  # Show full plan, not truncated
                "plan_steps": plan_steps,  # Also include as array for easier parsing
                "checks": checks,
                "retries": retries
            }
        }
```

**Explanation**:
- Assembles the final JSON response
- If no reasoning provided and status is success, generates a default message
- Returns structured JSON with:
  - `answer`: The final answer
  - `status`: "success" or "failed"
  - `reasoning_visible_to_user`: Human-readable explanation
  - `metadata`: Technical details (plan as string and array, checks, retry count)

**Example Success Output**:
```json
{
  "answer": "3 hours 35 minutes",
  "status": "success",
  "reasoning_visible_to_user": "Solved in 5 steps with 2 checks passing.",
  "metadata": {
    "plan": "1. Parse... | 2. Convert... | 3. Calculate... | 4. Convert... | 5. Format...",
    "plan_steps": ["1. Parse...", "2. Convert...", ...],
    "checks": [...],
    "retries": 0
  }
}
```

---

### 7. `interface.py` - User Interface

**Purpose**: Provides CLI and programmatic entry points.

```python
def cli():
    parser = argparse.ArgumentParser(description="Reasoning Agent CLI")
    parser.add_argument("--question", required=True, help="The word problem question")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    args = parser.parse_args()
    
    llm = LLMInterface(mock=args.mock)
    agent = ReasoningAgent(llm)
    result = agent.solve(args.question)
    print(json.dumps(result, indent=2))
```

**Explanation**:
- CLI function: parses command-line arguments
- `--question`: Required question string
- `--mock`: Optional flag to use mock LLM
- Creates LLM interface and agent, solves, prints JSON

**Usage**: `python3 interface.py --question "Your question" [--mock]`

```python
def solve(question: str, mock: bool = False) -> Dict[str, Any]:
    """Notebook-friendly function."""
    llm = LLMInterface(mock=mock)
    agent = ReasoningAgent(llm)
    return agent.solve(question)
```

**Explanation**:
- Programmatic entry point (for notebooks, scripts, API)
- Simple function interface
- Returns the result dictionary directly

**Usage in Python**:
```python
from interface import solve
result = solve("If a train leaves at 14:30...", mock=False)
```

---

### 8. `api.py` - REST API Server

**Purpose**: Exposes the agent as a web API using Flask.

```python
from flask import Flask, request, jsonify
from interface import solve
import os

app = Flask(__name__)

# Enable CORS if available (optional)
try:
    from flask_cors import CORS
    CORS(app)  # Enable CORS for all routes
except ImportError:
    # CORS not installed, continue without it
    pass
```

**Explanation**:
- Creates Flask app
- Optionally enables CORS (Cross-Origin Resource Sharing) for web browser access
- Gracefully handles missing CORS library

```python
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Multi-Step Reasoning Agent"})
```

**Explanation**:
- Simple health check endpoint
- Useful for monitoring and load balancers

```python
@app.route('/solve', methods=['POST'])
def solve_endpoint():
    try:
        # Get JSON payload
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON payload provided",
                "status": "error"
            }), 400
        
        # Extract question
        question = data.get('question')
        if not question:
            return jsonify({
                "error": "Missing 'question' field in payload",
                "status": "error"
            }), 400
        
        # Extract mock flag (optional, defaults to False)
        mock = data.get('mock', False)
        
        # Check if API key is set (unless mock mode)
        if not mock and not os.getenv('GOOGLE_API_KEY'):
            return jsonify({
                "error": "GOOGLE_API_KEY environment variable not set. Use mock=true for testing.",
                "status": "error"
            }), 500
```

**Explanation**:
- Main solve endpoint (POST only)
- Validates JSON payload exists
- Validates "question" field exists
- Extracts optional "mock" flag (defaults to False)
- Checks API key is set unless in mock mode
- Returns appropriate HTTP error codes (400 for bad request, 500 for server error)

```python
        # Solve the question
        result = solve(question, mock=mock)
        
        # Return the result
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
```

**Explanation**:
- Calls the `solve()` function from `interface.py`
- Returns result as JSON with 200 status
- Catches any exceptions and returns 500 error

```python
@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        "service": "Multi-Step Reasoning Agent API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "POST /solve": {
                "description": "Solve a word problem",
                "payload": {
                    "question": "string (required) - The word problem to solve",
                    "mock": "boolean (optional) - Use mock LLM, defaults to false"
                },
                "example": {
                    "question": "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
                    "mock": False
                }
            }
        }
    })
```

**Explanation**:
- Root endpoint provides API documentation
- Returns JSON describing all available endpoints
- Includes example payload

```python
if __name__ == '__main__':
    # Run on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
```

**Explanation**:
- Runs Flask development server
- `host='0.0.0.0'` allows access from any network interface
- `port=5000` is the default Flask port
- `debug=True` enables auto-reload and detailed error pages (disable in production)

---

## Complete Data Flow Example

Let's trace a complete example: **"If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"**

### Step 1: User calls `solve()` in `interface.py`
```python
solve("If a train leaves at 14:30 and arrives at 18:05, how long is the journey?", mock=False)
```

### Step 2: `agent.py` - Planning Phase
- Calls `plan(llm, question)`
- `planner.py` formats `PLANNER_PROMPT` with the question
- LLM returns: `{"steps": ["1. Parse times...", "2. Convert...", "3. Calculate...", "4. Format..."]}`
- Returns: `["1. Parse times...", "2. Convert...", "3. Calculate...", "4. Format..."]`

### Step 3: `agent.py` - Execution Phase
- Calls `execute(llm, question, plan_steps)`
- `executor.py` formats `EXEC_PROMPT` with question and plan
- LLM returns:
  ```json
  {
    "intermediates": {
      "step1": "Parsed: 14:30, 18:05",
      "step2": "Converted: 870 min, 1085 min",
      "step3": "Difference: 215 minutes",
      "step4": "Formatted: 3 hours 35 minutes"
    },
    "proposed_answer": "3 hours 35 minutes"
  }
  ```

### Step 4: `agent.py` - Verification Phase
- Calls `verify(llm, question, proposed)`
- `verifier.py`:
  - **LLM Check**: Formats `VERIFIER_PROMPT`, LLM independently solves and approves
  - **Code Check**: Detects "time" keyword, calls `_parse_time_diff()`:
    - Finds times: `["14:30", "18:05"]`
    - Parses: start=14:30, end=18:05
    - Calculates: 215 minutes (positive ✓)
    - Adds check: `{"check_name": "time_positive", "passed": True, "details": "Duration: 215 minutes"}`
- Returns:
  ```python
  {
    "approved": True,  # Both LLM and code checks passed
    "checks": [
      {"check_name": "time_positive", "passed": True, "details": "Duration: 215 minutes"},
      {"check_name": "llm_consistency", "passed": True, "details": "[]"}
    ],
    "issues": [],
    "re_solution": "14:30 to 18:05 is 3h35m"
  }
  ```

### Step 5: `agent.py` - Assemble Response
- Verification approved → returns success
- Calls `_assemble_json()`:
  ```json
  {
    "answer": "3 hours 35 minutes",
    "status": "success",
    "reasoning_visible_to_user": "Solved in 4 steps with 2 checks passing.",
    "metadata": {
      "plan": "1. Parse times... | 2. Convert... | 3. Calculate... | 4. Format...",
      "plan_steps": ["1. Parse times...", "2. Convert...", "3. Calculate...", "4. Format..."],
      "checks": [...],
      "retries": 0
    }
  }
  ```

### Step 6: Return to User
- `interface.py` returns the JSON
- If called via CLI: prints to stdout
- If called via API: returns HTTP 200 with JSON body

---

## Retry Logic Flow

If verification fails:

1. **First Attempt** (retries=0):
   - Plan → Execute → Verify → **FAILED**
   - Increment retries to 1
   - Log warning with issues

2. **Second Attempt** (retries=1):
   - Plan → Execute → Verify → **FAILED**
   - Increment retries to 2
   - Log warning with issues

3. **Third Attempt** (retries=2):
   - Plan → Execute → Verify → **FAILED**
   - retries (2) > max_retries (2) → Return failure JSON

**Note**: Each retry regenerates the plan, so the agent can try different approaches.

---

## Key Design Decisions

1. **JSON-only outputs**: Makes parsing reliable, avoids free-text parsing errors
2. **Low temperature (0.1)**: More deterministic, consistent outputs
3. **Three-phase approach**: Separates concerns (planning vs execution vs verification)
4. **Dual verification**: LLM re-solves + code-based checks catch different error types
5. **Retry mechanism**: Handles transient failures and verification issues
6. **Mock mode**: Enables testing without API costs
7. **Structured logging**: All phases logged to `agent.log` for debugging
8. **Graceful error handling**: Always returns valid JSON, never crashes

---

## Testing

- **`test_example.py`**: Simple test script
- **`tests/test_agent.py`**: Pytest suite with multiple test cases
- **`test_api.py`**: Tests the API endpoints
- **`run_tests.py`**: Runs all tests and generates logs

---

## Summary

This project implements a robust, multi-phase reasoning agent that:
- Plans solutions step-by-step
- Executes calculations following the plan
- Verifies solutions with dual checks (LLM + code)
- Retries on failures
- Provides structured JSON outputs
- Supports both CLI and API interfaces
- Includes comprehensive error handling and logging

The code is modular, well-separated, and designed for extensibility (e.g., adding new verification checks, different LLM providers, etc.).

