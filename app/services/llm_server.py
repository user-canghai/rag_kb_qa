"""本地 NPU 大模型服务：提供 OpenAI 兼容接口。"""
import importlib
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import time
import uuid
import threading
from typing import List, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app import config
from app.core.progress import banner, fmt_seconds, info, kv, ok, section

importlib.import_module("torch_npu")

MODEL_PATH = str(config.LLM_PATH)
DEVICE = config.LLM_DEVICE
API_KEY = config.LLM_SERVER_KEY
SERVED_MODEL = config.MODEL_NAME

app = FastAPI(title="Local NPU LLM (OpenAI compatible)")
_lock = threading.Lock()
_cache = {}


def load_model():
    if "model" not in _cache:
        section("加载本地大模型到 NPU")
        info(f"模型路径：{MODEL_PATH}")
        info(f"推理设备：{DEVICE}（首次加载需要几十秒，请等待）")
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        info(f"tokenizer 就绪（{fmt_seconds(time.time() - started)}）")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(DEVICE)
        model.eval()
        _cache["tokenizer"] = tokenizer
        _cache["model"] = model
        ok(f"模型加载完成，总用时 {fmt_seconds(time.time() - started)}")
    return _cache["tokenizer"], _cache["model"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = SERVED_MODEL
    messages: List[dict]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    response_format: Optional[dict] = None


def build_inputs(tokenizer, messages, response_format):
    msgs = list(messages)
    if response_format and response_format.get("type") == "json_object":
        msgs.append({"role": "system", "content": "只输出一个合法的 JSON 对象，不要输出任何其他内容。"})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").to(DEVICE)


def chunk_payload(content, finish_reason=None):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": SERVED_MODEL,
        "choices": [{"index": 0, "delta": ({"content": content} if content else {}), "finish_reason": finish_reason}],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "loaded": "model" in _cache}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL, "object": "model", "owned_by": "local-npu"}]}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest, authorization: str = Header(None)):
    if authorization and not authorization.endswith(API_KEY):
        raise HTTPException(status_code=403, detail="invalid api key")

    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer, request.messages, request.response_format)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    info(
        f"收到请求：{len(request.messages)} 条消息 / {prompt_tokens} tokens，"
        f"stream={request.stream}，max_tokens={request.max_tokens}"
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=request.temperature > 0,
        temperature=max(request.temperature, 1e-5),
        top_p=request.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    if not request.stream:
        started = time.time()
        with _lock, torch.no_grad():
            out = model.generate(**gen_kwargs)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - started
        ok(
            f"生成完成：{len(text)} 字，用时 {fmt_seconds(elapsed)}，"
            f"约 {len(text) / elapsed:.1f} 字/秒"
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": 0, "total_tokens": 0},
        }

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    def run():
        with _lock, torch.no_grad():
            model.generate(**gen_kwargs)

    def event_stream():
        started = time.time()
        chars = 0
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        yield "data: " + json.dumps(chunk_payload(""), ensure_ascii=False) + "\n\n"
        for piece in streamer:
            if piece:
                chars += len(piece)
                yield "data: " + json.dumps(chunk_payload(piece), ensure_ascii=False) + "\n\n"
        yield "data: " + json.dumps(chunk_payload(None, "stop"), ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"
        thread.join()
        elapsed = time.time() - started
        ok(
            f"流式生成结束：{chars} 字，用时 {fmt_seconds(elapsed)}，"
            f"约 {chars / elapsed:.1f} 字/秒（含首 token 等待）"
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    banner(
        "本地 NPU 大模型服务（OpenAI 兼容）",
        "对外模型名 qwen3.5-plus；模型在第一个请求到达时才真正加载到 NPU",
    )
    kv(
        [
            ("模型路径", MODEL_PATH),
            ("推理设备", DEVICE),
            ("监听端口", config.LLM_PORT),
            ("健康检查", f"http://127.0.0.1:{config.LLM_PORT}/health"),
        ]
    )
    uvicorn.run(app, host="0.0.0.0", port=config.LLM_PORT)
