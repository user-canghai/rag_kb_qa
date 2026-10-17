"""答案自我反思：校验回答是否完全来自检索上下文。"""
import json
import time

from app import config
from app.core.llm import get_client
from app.core.progress import fmt_seconds, info, ok

PROMPT = """你是RAG系统专业反思器。任务：严格检查回答是否完全来自参考资料，不允许编造。

参考资料：
{context}

用户问题：
{query}

回答：
{answer}

请检查以下项：
1. 回答是否全部来自参考资料？
2. 是否存在幻觉、捏造、无依据内容？
3. 是否准确、不偏离问题？

只返回格式（JSON）：
{{"pass": true/false, "reason": "原因"}}"""


def reflect_answer(query: str, context: str, answer: str, client=None) -> dict:
    """校验答案是否忠于参考资料。

    返回约定：
    - {"pass": True,  "reason": ...}  通过
    - {"pass": False, "reason": ...}  未通过（内容问题）
    - {"pass": None,  "reason": ...}  反思调用失败（模块异常，区别于未通过）
    """
    started = time.time()
    info(f"反思输入：参考资料 {len(context)} 字，待校验回答 {len(answer)} 字")
    prompt = PROMPT.format(context=context, query=query, answer=answer)
    client = client or get_client()
    try:
        resp = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        ok(f"反思结果：{result}（用时 {fmt_seconds(time.time() - started)}）")
        return {"pass": bool(result.get("pass")), "reason": str(result.get("reason", ""))}
    except Exception as e:
        print("反思校验失败:", e)
        return {"pass": None, "reason": f"反思调用失败：{e}"}


def reflect_stream(query, context_text, answer, client=None):
    """把反思结果以 type=reflection 的流式消息注入前端。"""
    info("开始反思：校验答案是否完全来自参考资料")
    yield json.dumps(
        {"content": "\n\n        正在反思答案...", "type": "reflection", "status": "pending"},
        ensure_ascii=False,
    )
    res = reflect_answer(query, context_text, answer, client)
    if res["pass"] is True:
        msg = "\n\n✅ 答案已通过反思校验"
    elif res["pass"] is False:
        msg = f"\n\n❌ 答案未通过反思:{res['reason']}"
    else:
        msg = f"\n\n⚠️ 反思模块不可用:{res['reason']}"
    yield json.dumps({"content": msg, "type": "reflection", "status": "done"}, ensure_ascii=False)
