PLANNER_PROMPT = """
You are a precise planner for solving word problems. Given a question, output a JSON with a numbered list of 3-6 concise steps to solve it. Steps should cover parsing, computation, validation, and formatting. 
Example: {{"steps": ["1. Parse quantities: extract numbers and relations.", "2. Compute totals: add reds + greens.", "3. Validate: ensure non-negative.", "4. Format answer."]}}

Question: {question}
Output only JSON.
"""

EXEC_PROMPT = """
You are an executor following a strict plan. For the question and plan, perform each step in order. Show intermediate results as JSON: {{"intermediates": {{"step1": "result1", "step2": "result2"}}, "proposed_answer": "short final answer"}}. Use arithmetic/code-like notation for calcs. Do not skip steps.

Question: {question}
Plan: {plan_steps}
Output only JSON.
"""

VERIFIER_PROMPT = """
You are a verifier checking a proposed solution for a question. Independently re-solve briefly, then check: (1) Matches proposed? (2) Constraints valid (e.g., positive nums, time logic)? (3) Consistent explanation? Output JSON: {{"approved": true/false, "issues": ["list of problems"], "re_solution": "brief alt solve"}}.

Question: {question}
Proposed: {proposed_json}
Output only JSON.
"""

