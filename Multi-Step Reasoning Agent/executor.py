import json
from typing import Dict, Any, List
from prompts import EXEC_PROMPT
from llm_interface import LLMInterface

def execute(llm: LLMInterface, question: str, plan_steps: List[str]) -> Dict[str, Any]:
    prompt = EXEC_PROMPT.format(question=question, plan_steps="\n".join(plan_steps))
    response = llm.call(prompt, system="You are an executor following a strict plan.")
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

