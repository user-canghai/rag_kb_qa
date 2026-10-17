"""混合检索：FAISS 向量检索（0.6）+ BM25 关键词检索（0.4）。

BM25 检索器按库缓存（state.bm25），入库 / 删除 / 切库后由 vector_store 失效，
避免每次查询都从零重建。
"""
import time

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app import config
from app.core.progress import fmt_seconds, info, ok
from app.state import state


def get_bm25_from_faiss(faiss_db, k: int = 3):
    """构建 BM25 检索器（带缓存）：同一索引实例 + 同一 k 直接复用。"""
    cached = state.bm25
    if cached is not None and cached[0] == id(faiss_db) and cached[1] == k:
        return cached[2]
    all_docs = list(faiss_db.docstore._dict.values())
    started = time.time()
    info(f"构建 BM25 关键词检索器：{len(all_docs)} 个文本块，k={k}")
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k
    info(f"BM25 就绪，用时 {fmt_seconds(time.time() - started)}")
    state.bm25 = (id(faiss_db), k, bm25_retriever)
    return bm25_retriever


def create_hybrid_retriever(faiss_db, weight_faiss=None, weight_bm25=None, k=None):
    """向量 + BM25 的加权融合检索器（权重与 k 默认取 config）。"""
    weight_faiss = config.HYBRID_FAISS_WEIGHT if weight_faiss is None else weight_faiss
    weight_bm25 = config.HYBRID_BM25_WEIGHT if weight_bm25 is None else weight_bm25
    k = config.RETRIEVAL_K if k is None else k
    info(f"创建混合检索器：向量权重 {weight_faiss} / BM25 权重 {weight_bm25}，各路召回 top {k}")
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": k})
    bm25_retriever = get_bm25_from_faiss(faiss_db, k=k)
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[weight_faiss, weight_bm25],
    )


def hybrid_search(faiss_db, query, k=None):
    """一次混合检索：返回重排前的候选子块列表。"""
    k = config.RETRIEVAL_K if k is None else k
    started = time.time()
    docs = create_hybrid_retriever(faiss_db, k=k).invoke(query)
    ok(f"混合检索完成：召回 {len(docs)} 个子块，用时 {fmt_seconds(time.time() - started)}")
    return docs
