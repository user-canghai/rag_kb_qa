"""RAG 主编排：检索增强问答、普通对话与对外简化链路。

- generate_rag：完整链路（HyDE → 混合检索 → 重排 → 置信度过滤 → 生成 → 反思）
- generate_plain：不检索的普通对话
- basic_rag_with_stream：供 api_server 使用的简化链路（教学用最小闭环）
- stream_and_cache：统一的流式包装（查询改写、写对话历史与问答缓存）
"""
import json

from app import config
from app.core.llm import get_client
from app.core.progress import Steps, StreamMeter, info, section
from app.data import vector_store
from app.generation.rewrite import rewrite_with_history
from app.state import state


def generate_rag(query):
    """完整 RAG 链路：逐段 yield JSON 行（前端按行解析）。"""
    steps = Steps("RAG 检索增强问答")
    steps.begin("加载向量模型与索引")
    vector_store.ensure_db()
    faiss_db = state.faiss_db
    if not faiss_db:
        steps.end("索引为空")
        steps.finish("已中断")
        print("请先上传文档构建索引")
        yield json.dumps({"content": "请先上传文档构建索引"}, ensure_ascii=False)
        return
    steps.end(f"{vector_store.current_kb()}，共 {faiss_db.index.ntotal} 块")

    from app.retrieval.hybrid import hybrid_search
    from app.retrieval.hyde import generate_hyde_query

    steps.begin("HyDE 假设文档增强")
    if config.ENABLE_HYDE:
        hyde_query = generate_hyde_query(query)
        steps.note(f"原问题：{query[:60]}")
        steps.note(f"假答案：{hyde_query[:60]}{'...' if len(hyde_query) > 60 else ''}")
        steps.end(f"共 {len(hyde_query)} 字")
    else:
        hyde_query = query
        steps.note("ENABLE_HYDE=0，直接使用原问题检索")
        steps.end("已跳过")

    steps.begin("混合检索（向量 + BM25）")
    docs = hybrid_search(faiss_db, hyde_query, k=config.RETRIEVAL_K)
    steps.end(f"召回 {len(docs)} 个子块")

    from app.retrieval.rerank import rerank_documents_with_score

    steps.begin("CrossEncoder 重排序")
    docs = rerank_documents_with_score(query, docs)
    scores = [doc.metadata.get("score") for doc in docs]
    if scores:
        steps.end("得分 " + "、".join(str(score) for score in scores))
    else:
        steps.end("没有候选子块")

    from app.retrieval.confidence import filter_by_confidence, has_high_confidence

    steps.begin(f"置信度过滤（阈值 {config.MIN_CONFIDENCE}）")
    docs = filter_by_confidence(docs, min_score=config.MIN_CONFIDENCE)
    steps.end(f"剩余 {len(docs)} 个子块")
    if not has_high_confidence(docs, min_score=config.MIN_CONFIDENCE):
        steps.note("没有高置信度素材，改为使用通用知识回答")
        yield json.dumps(
            {"retrieval": [{"content": "未找到高置信度的相关内容，将使用通用知识回答", "source": "系统提示"}]},
            ensure_ascii=False,
        )

    steps.begin("整理检索来源")
    retrieval = []
    exist_items = set()
    for doc in docs:
        content = (
            doc.metadata["parent_content"]
            if config.CHUNK_METHOD == "parent_child"
            else doc.page_content
        )
        source = doc.metadata.get("filename", doc.metadata.get("file", "未知来源"))
        key = (content, source)
        if key in exist_items:
            continue
        retrieval.append({"content": content, "source": source})
        exist_items.add(key)
    steps.end(f"{len(retrieval)} 条来源")

    if retrieval:
        yield json.dumps({"retrieval": retrieval}, ensure_ascii=False)

    context_text = "\n".join(d["content"] for d in retrieval)

    steps.begin("大模型流式生成")
    meter = StreamMeter("生成中")
    resp = get_client().chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": f"上下文：{context_text}\n问题：{query}"}],
        stream=True,
    )

    full_answer = ""
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            meter.update(len(text))
            yield json.dumps({"content": text}, ensure_ascii=False)
    meter.finish(f"上下文 {len(context_text)} 字")
    steps.end(f"回答 {len(full_answer)} 字")

    from app.generation.reflection import reflect_stream

    steps.begin("答案自我反思")
    for res in reflect_stream(query, context_text, full_answer):
        yield res
    steps.end()

    steps.finish(f"回答 {len(full_answer)} 字")


def generate_plain(query):
    """不检索知识库的普通对话：仅流式生成。"""
    steps = Steps("普通对话（不检索知识库）")
    steps.begin("大模型流式生成")
    meter = StreamMeter("生成中")
    resp = get_client().chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": query}],
        stream=True,
    )
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            meter.update(len(text))
            yield json.dumps({"content": text}, ensure_ascii=False)
    meter.finish()
    steps.end()
    steps.finish()


async def basic_rag_with_stream(query):
    """供 api_server 调用的简化 RAG 链路（无 HyDE / 重排 / 过滤 / 反思）。"""
    from app.retrieval.hybrid import hybrid_search

    vector_store.ensure_db()
    faiss_db = state.faiss_db
    if not faiss_db:
        yield json.dumps({"content": "请先上传文档构建索引"}, ensure_ascii=False)
        return

    docs = hybrid_search(faiss_db, query, k=config.RETRIEVAL_K)
    context_text = "\n".join(d.page_content for d in docs[:3])
    print(f"检索结果：{context_text[:200]}")

    resp = get_client().chat.completions.create(
        model=config.MODEL_NAME,
        messages=[{"role": "user", "content": f"上下文：{context_text}\n问题：{query}"}],
        stream=True,
    )
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            yield json.dumps({"content": chunk.choices[0].delta.content}, ensure_ascii=False)


def stream_and_cache(query, mode, cache_key, gen_func):
    """一轮问答的完整流式包装：改写（仅 RAG）→ 生成 → 写历史与缓存。"""
    section(f"收到提问（模式：{mode}）")
    info(f"原始问题：{query}")
    if mode == "rag":
        rewrite_query = rewrite_with_history(query)
        info(f"用于检索的问题：{rewrite_query}")
    else:
        rewrite_query = query

    full_text = ""
    raw_stream = ""
    retrieval_data = []
    for data_str in gen_func(rewrite_query):
        chunk = data_str + "\n"
        raw_stream += chunk
        yield chunk
        try:
            data = json.loads(data_str)
        except Exception:
            continue
        if "content" in data and data.get("type") != "reflection":
            full_text += data["content"]
        if "retrieval" in data:
            retrieval_data = data["retrieval"]

    state.chat_history.append(
        {"question": query, "answer": full_text, "mode": mode, "retrieval": retrieval_data}
    )
    state.qa_cache.put(cache_key, raw_stream)
    info(f"结果已写入问答缓存（当前 {len(state.qa_cache)}/{config.MAX_CACHE_SIZE} 条）")
