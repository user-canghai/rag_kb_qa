"""基于 FastAPI 的流式 RAG 问答服务，供智能体作为工具调用。"""
import asyncio
import time
import warnings

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config
from app.core.progress import banner, fmt_seconds, kv, ok, section, warn
from app.generation.pipeline import basic_rag_with_stream

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG 流式问答接口")

VALID_API_KEY = config.SERVICE_API_KEY
API_PORT = config.API_PORT


class QueryRequest(BaseModel):
    question: str


@app.post("/api/rag")
async def rag_stream_api(request: QueryRequest, authorization: str = Header(None)):
    started = time.time()
    if not VALID_API_KEY:
        warn("拒绝请求：服务未配置 RAG_API_KEY，返回 503")
        raise HTTPException(status_code=503, detail="服务未配置鉴权密钥")
    if not authorization or not authorization.startswith("Bearer "):
        warn("拒绝请求：缺少 Bearer 鉴权头，返回 401")
        raise HTTPException(status_code=401, detail="未授权或格式错误")
    if authorization.split(" ")[1] != VALID_API_KEY:
        warn("拒绝请求：密钥无效，返回 403")
        raise HTTPException(status_code=403, detail="秘钥无效")

    section(f"收到 RAG 调用：{request.question[:40]}")

    async def stream_generator():
        chars = 0
        async for chunk in basic_rag_with_stream(request.question):
            chars += len(chunk)
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)
        ok(f"SSE 推送结束：{chars} 字节，总耗时 {fmt_seconds(time.time() - started)}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    banner(
        "RAG 服务化接口（FastAPI + SSE）",
        "供上层智能体作为工具调用，请求需携带 Authorization: Bearer <密钥>",
    )
    kv(
        [
            ("接口地址", f"http://<服务器IP>:{API_PORT}/api/rag"),
            ("监听端口", API_PORT),
            ("提示", "API 端口默认为 7862，避免与页面服务冲突"),
        ]
    )
    uvicorn.run("app.services.api:app", host="0.0.0.0", port=API_PORT)
