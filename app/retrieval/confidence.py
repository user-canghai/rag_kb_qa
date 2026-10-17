"""按重排序置信度过滤低质量检索结果。"""


def filter_by_confidence(docs, min_score: float = 0.5):
    return [doc for doc in docs if doc.metadata.get("score", 1.0) >= min_score]


def has_high_confidence(docs, min_score: float = 0.5) -> bool:
    return any(doc.metadata.get("score", 1.0) >= min_score for doc in docs)
