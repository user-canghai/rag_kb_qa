"""重排序：交叉编码器对候选子块打分排序，并提供带分数的版本用于置信度过滤。"""
import time

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app import config
from app.core.progress import fmt_seconds, info, ok

_rerank_model = None


def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        info(
            f"加载重排序模型：{config.RERANK_PATH}"
            f"（设备 {config.RERANK_DEVICE}，首次加载较慢）"
        )
        started = time.time()
        _rerank_model = HuggingFaceCrossEncoder(
            model_name=str(config.RERANK_PATH),
            model_kwargs={"device": config.RERANK_DEVICE},
        )
        ok(f"重排序模型就绪，用时 {fmt_seconds(time.time() - started)}")
    return _rerank_model


def get_reranker(top_n: int | None = None):
    top_n = config.RERANK_TOP_N if top_n is None else top_n
    return CrossEncoderReranker(model=get_rerank_model(), top_n=top_n)


def rerank_documents(query, docs):
    return get_reranker().compress_documents(docs, query)


def rerank_documents_with_score(query, docs, top_n: int | None = None):
    top_n = config.RERANK_TOP_N if top_n is None else top_n
    if not docs:
        info("没有候选子块，跳过重排序")
        return docs
    started = time.time()
    pairs = [[query, doc.page_content] for doc in docs]
    info(f"重排序打分：{len(pairs)} 个（问题, 子块）组合")
    scores = get_rerank_model().score(pairs)
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = round(float(score), 4)
    ranked = sorted(docs, key=lambda d: d.metadata.get("score", 0), reverse=True)[:top_n]
    ok(
        f"重排序完成：{len(docs)} -> {len(ranked)} 块，"
        f"最高分 {ranked[0].metadata['score'] if ranked else '-'}，"
        f"用时 {fmt_seconds(time.time() - started)}"
    )
    return ranked
