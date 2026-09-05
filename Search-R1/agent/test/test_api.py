from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

### deepseek系列
# messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
# response = client.chat.completions.create(
#     model="deepseek-v3.2-exp",  # deepseek-v3-0324，deepseek-v3.1，deepseek-v3.2-exp，deepseek-r1-0528
#     messages=messages,
#     extra_body={"thinking": {"type": "enabled"}} # enabled/disabled
# )

# print("模型原始输出", response)

### qwen系列
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="qwen3-32b",  # qwen3-32b, qwen3-max, qwen3-235b-a22b-thinking-2507, qwen3-235b-a22b-instruct-2507
    messages=messages,
    extra_body={"enable_thinking":False}, # True/False
)

print("模型原始输出", response)
