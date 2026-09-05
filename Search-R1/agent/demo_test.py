import os
from types import SimpleNamespace

from openai import OpenAI
from agent.llm_agent import AgentConfig, SearchAgent
from evaluate.evaluate_hotpot_2wiki import noise_config_preparation
from noise_bench.noiser.factory import NoiserFactory

if __name__ == "__main__":
    """
    示例：
    OPENAI_API_KEY=xxx OPENAI_BASE_URL=https://api.xxx/v1 \
      python demo_test.py
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    cfg = AgentConfig(
        model=os.getenv("AGENT_MODEL", "deepseek-r1-0528"),
        max_turns=5,
        max_output_tokens=512,
        temperature=0.7,
        verbose=True,
    )
    
    args = {
        "noise_type": "redundancy",
        "noise_model": "gpt-4o-mini",
        "language": "zh",
        "apply_always": True,
        "start_step": 1,
        "prob_per_step": 0.5,
        "interval_steps": 1,
        "max_times_per_trail": 3,
    }
    
    # 将 dict 转为带属性的对象，传入 noise_config_preparation
    noise_args = SimpleNamespace(**args)
    noise_cfg = noise_config_preparation(noise_args)
    
    noiser = NoiserFactory.get_noiser_instance(noise_cfg)

    # 这里只是 demo，真实实验时请把 noiser 实例传进来
    agent = SearchAgent(client=client, config=cfg, noiser=noiser,searcher=None)

    q = "Which US state is the birthplace of the author of 'The Shining'?"
    result = agent.run(q)
    print("\n[RESULT]")
    print("Question:", result.question)
    print("Final answer:", result.final_answer)
    print("Raw model answer:", result.raw_model_answer)
    print("Steps:", result.steps, "finish_reason:", result.finish_reason)