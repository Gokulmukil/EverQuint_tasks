import google.generativeai as genai
import os
import json
from typing import Dict, Any

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
        
        self.mock_responses = {
            "planner": '{"steps": ["1. Parse the question.", "2. Extract key numbers.", "3. Perform calculation.", "4. Validate result.", "5. Format answer."]}',
            "executor": '{"intermediates": {"step1": "parsed: times 14:30 to 18:05", "step2": "diff: 3h35m", "step3": "valid", "step4": "ok"}, "proposed_answer": "3 hours 35 minutes"}',
            "verifier": '{"approved": true, "issues": [], "re_solution": "14:30 to 18:05 is 3h35m"}'
        }  # Simple mocks; extend as needed

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
        
        try:
            # Combine system and user prompt for Gemini
            full_prompt = f"{system}\n\n{prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.95,
                    top_k=40,
                )
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

