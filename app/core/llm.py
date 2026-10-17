"""OpenAI 兼容客户端：统一指向本地 NPU 大模型服务（llm_server）。

全项目（RAG 生成、查询改写、HyDE、摘要、反思）共用同一个客户端实例。
"""
import openai

from app import config

_client = None


def get_client() -> openai.OpenAI:
    """进程内单例：所有调用都走 .env 里的 BASE_URL / QWEN_API_KEY。"""
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=config.LLM_CLIENT_KEY, base_url=config.BASE_URL)
    return _client
