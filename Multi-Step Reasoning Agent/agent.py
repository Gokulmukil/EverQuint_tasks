import json
import logging
from typing import Dict, Any, List
from planner import plan
from executor import execute
from verifier import verify

logging.basicConfig(filename='agent.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class ReasoningAgent:
    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

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

                if verification["approved"]:
                    return self._assemble_json(proposed["proposed_answer"], "success", plan_steps, verification["checks"], retries)
                else:
                    retries += 1
                    logging.warning(f"Retry {retries}: Issues - {verification['issues']}")
                    if retries > self.max_retries:
                        short_reasoning = f"Failed after {retries} retries: {', '.join(verification['issues'][:2])}"
                        return self._assemble_json(proposed.get("proposed_answer", "N/A"), "failed", plan_steps, verification["checks"], retries, short_reasoning)
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

        # Fallback (unreachable)
        return {"answer": "Failed", "status": "failed", "reasoning_visible_to_user": "Max retries exceeded", "metadata": {"retries": retries}}

    def _assemble_json(self, answer: str, status: str, plan_steps: List[str], checks: List[Dict], retries: int, reasoning: str = "") -> Dict[str, Any]:
        if not reasoning and status == "success":
            reasoning = f"Solved in {len(plan_steps)} steps with {len(checks)} checks passing."
        return {
            "answer": answer,
            "status": status,
            "reasoning_visible_to_user": reasoning,
            "metadata": {
                "plan": plan_steps,  # Array of plan steps for easier parsing and display
                "checks": checks,
                "retries": retries
            }
        }

