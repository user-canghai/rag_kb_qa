"""多轮对话查询改写：结合最近历史把当前问题改写为独立问句。"""
import time

from app import config
from app.core.llm import get_client
from app.core.progress import fmt_seconds, info
from app.state import state


def rewrite_with_history(query: str, client=None) -> str:
    """结合最近 N 轮历史改写问题；无历史或改写失败时返回原问题。"""
    if not state.chat_history:
        info("没有历史对话，直接使用原问题")
        return query

    rounds = config.HISTORY_REWRITE_ROUNDS
    info(f"结合最近 {min(len(state.chat_history), rounds)} 轮历史改写问题")
    history_prompt = "".join(
        f"用户：{h['question']}\n助手：{h['answer'][:200]}\n" for h in state.chat_history[-rounds:]
    )
    prompt = f"""你是RAG检索优化助手。
根据历史对话，把当前用户的问题改写为完整、独立、无指代的问题。
只输出改写后的问题，不要多余内容。
如果历史对话与当前问题毫无关联，请忽略历史对话内容。
历史对话：
{history_prompt}

当前问题：{query}

改写后："""

    client = client or get_client()
    started = time.time()
    try:
        resp = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=128,
        )
        rewritten = resp.choices[0].message.content.strip()
        info(f"改写完成（{fmt_seconds(time.time() - started)}）：{rewritten[:50]}")
        return rewritten
    except Exception as e:
        print(f"查询改写失败，使用原问题：{e}")
        return query
