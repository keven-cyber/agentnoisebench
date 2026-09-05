"""Small example showing how to instantiate and run the SearchAgent.

This file can be executed as a quick smoke-test. It expects the environment
variables OPENAI_BASE_URL (required) and OPENAI_API_KEY 
to be set, and a local retrieval service listening at http://127.0.0.1:8008/retrieve.

Usage (from repository root):
    python3 -m agent.run_example
"""
import os
from .agent import SearchAgent
from .llm import LLMClient
from .searcher import Searcher


def main():
    llm = LLMClient()
    searcher = Searcher()
    agent = SearchAgent(llm_client=llm, searcher=searcher)

    q = (
        "Mike Barnett negotiated many contracts including which player that went on to become "
        "general manager of CSKA Moscow of the Kontinental Hockey League?"
    )
    print("Question:\n", q)
    ans = agent.run(q)
    print("\nAnswer:\n", ans)


if __name__ == "__main__":
    main()
