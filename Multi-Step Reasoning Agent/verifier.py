import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
from prompts import VERIFIER_PROMPT
from llm_interface import LLMInterface

def verify(llm: LLMInterface, question: str, proposed: Dict[str, Any]) -> Dict[str, Any]:
    # LLM check
    prompt = VERIFIER_PROMPT.format(question=question, proposed_json=json.dumps(proposed))
    try:
        response = llm.call(prompt, system="You are a verifier checking a proposed solution.")
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
        
        llm_check = json.loads(response_clean)
    except json.JSONDecodeError as e:
        llm_check = {"approved": False, "issues": [f"JSON parse error: {e}"], "re_solution": ""}

    # Code-based checks
    checks = []
    if "time" in question.lower() or "train" in question.lower() or "meeting" in question.lower() or "slot" in question.lower():
        duration = _parse_time_diff(question)
        checks.append({
            "check_name": "time_positive",
            "passed": duration > 0,
            "details": f"Duration: {duration} minutes"
        })
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
            else:
                checks.append({"check_name": "non_negative", "passed": True, "details": "Could not parse total, but no negative found"})
        except:
            checks.append({"check_name": "non_negative", "passed": True, "details": "Could not parse total"})

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
    # Also check for time ranges like "09:00–09:30"
    ranges = re.findall(r'(\d{2}:\d{2})[–-](\d{2}:\d{2})', question)
    if ranges:
        try:
            start_str, end_str = ranges[0]
            start = datetime.strptime(start_str, '%H:%M')
            end = datetime.strptime(end_str, '%H:%M')
            if end < start:
                end += timedelta(days=1)
            delta = end - start
            return int(delta.total_seconds() / 60)
        except:
            pass
    return 0

