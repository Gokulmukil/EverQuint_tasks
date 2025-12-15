import argparse
import json
from typing import Dict, Any
from llm_interface import LLMInterface
from agent import ReasoningAgent

def cli():
    parser = argparse.ArgumentParser(description="Reasoning Agent CLI")
    parser.add_argument("--question", required=True, help="The word problem question")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    args = parser.parse_args()
    
    llm = LLMInterface(mock=args.mock)
    agent = ReasoningAgent(llm)
    result = agent.solve(args.question)
    print(json.dumps(result, indent=2))

def solve(question: str, mock: bool = False) -> Dict[str, Any]:
    """Notebook-friendly function."""
    llm = LLMInterface(mock=mock)
    agent = ReasoningAgent(llm)
    return agent.solve(question)

if __name__ == "__main__":
    cli()

