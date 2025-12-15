import json
from typing import List
from prompts import PLANNER_PROMPT
from llm_interface import LLMInterface

def plan(llm: LLMInterface, question: str) -> List[str]:
    prompt = PLANNER_PROMPT.format(question=question)
    response = llm.call(prompt, system="You are a precise planner for solving word problems.")
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
        return data["steps"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid plan JSON: {e}. Response was: {response}")

