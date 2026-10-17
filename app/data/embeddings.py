"""句向量模型：fastembed(ONNX) 的 LangChain 适配与进程内懒加载。"""
import time
from typing import List

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from app import config
from app.core.progress import fmt_seconds, info, ok
from app.state import state


class FastEmbedEmbeddings(Embeddings):
    """fastembed(ONNX) 的 LangChain 适配器，模型离线缓存在 TEXT_EMBED_PATH。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model_instance = TextEmbedding(
            model_name=model_name,
            cache_dir=str(config.TEXT_EMBED_PATH),
            local_files_only=True,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self._model_instance.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(list(self._model_instance.embed(text))[0])

    @property
    def model(self):
        return self.model_name


def ensure_embed_model() -> FastEmbedEmbeddings:
    """懒加载句向量模型（首次调用较慢），进程内单例。"""
    if state.embed_model is None:
        info(f"加载 Embedding 模型：{config.TEXT_EMBED_PATH}（首次加载较慢，请等待）")
        started = time.time()
        state.embed_model = FastEmbedEmbeddings()
        ok(f"Embedding 模型就绪，用时 {fmt_seconds(time.time() - started)}")
    return state.embed_model
