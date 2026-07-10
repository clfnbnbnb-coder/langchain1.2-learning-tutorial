"""
@author:LingFeng Chen
@description:tool_invoke_test
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from rich import print as rprint

# 1、提供大模型
load_dotenv(override=True)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    # temperature=1.5,
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    # max_tokens=10,
)
# 2、声明一个函数
def get_weather(city: str):
    return f"{city}天气晴朗~~"

# 3、将函数绑定在模型上
model_with_tools = model.bind_tools([get_weather])

# 4、调用模型
response = model_with_tools.invoke("仲恺的天气怎么样？")
rprint(response)