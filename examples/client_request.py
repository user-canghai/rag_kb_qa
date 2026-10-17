"""RAG 服务调用示例：SSE 流式接收并打印答案。

运行：python -m examples.client_request
"""
import json
import os
import time

import requests

from app import config
from app.core.progress import banner, fmt_seconds, info, ok

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:7862/api/rag")
API_KEY = config.SERVICE_API_KEY


def ask_rag_stream(question):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    started = time.time()
    pieces = 0
    chars = 0
    try:
        info(f"向 {API_URL} 发起流式请求")
        response = requests.post(
            API_URL, json={"question": question}, headers=headers, stream=True, timeout=120
        )
        response.raise_for_status()
        info(f"HTTP {response.status_code}，开始接收 SSE 分片")
        print("AI回答：", end="", flush=True)
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode("utf-8")
                if chunk.startswith("data: "):
                    content = json.loads(chunk[6:])["content"]
                    pieces += 1
                    chars += len(content)
                    print(content, end="", flush=True)
        print()
        ok(
            f"接收完成：{pieces} 个分片 / {chars} 字，"
            f"总耗时 {fmt_seconds(time.time() - started)}"
        )
    except Exception as e:
        print(f"错误：{e}")


if __name__ == "__main__":
    banner("client_request.py - RAG 接口调用客户端", API_URL)
    question = "IPRAN设备出现LAG告警应该怎么处理？"
    print(f"问题：{question}")
    ask_rag_stream(question)
