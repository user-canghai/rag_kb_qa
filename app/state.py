import threading
from pathlib import Path
from typing import Optional

from app import config
from app.core.cache import LRUCache


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.embed_model = None
        self.faiss_db = None
        self.index_dir: Optional[Path] = None
        self.bm25 = None
        self.chat_history = []
        self.qa_cache = LRUCache(config.MAX_CACHE_SIZE)
        self.ragas_result = {}
        self.ragas_progress = {"status": "等待评估", "progress": 0, "current": 0, "total": 0}

    def reset_database(self) -> None:
        """索引与 BM25 缓存失效（切库、清空、索引内容变更后调用）。"""
        self.faiss_db = None
        self.bm25 = None

    def reset_ragas(self) -> None:
        self.ragas_result.clear()
        self.ragas_progress.update(
            {"status": "等待评估", "progress": 0, "current": 0, "total": 0}
        )

    def clear_conversation(self) -> None:
        self.chat_history = []
        self.qa_cache.clear()


state = RuntimeState()
