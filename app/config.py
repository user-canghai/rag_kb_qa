import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_name: str, default: str) -> Path:
    """路径配置：相对路径相对于项目根目录解析为绝对路径。"""
    raw = os.getenv(env_name, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _bool(env_name: str, default: bool) -> bool:
    return os.getenv(env_name, "1" if default else "0") not in ("0", "false", "False")


LLM_SERVER_KEY = os.getenv("LLM_API_KEY", "")
LLM_CLIENT_KEY = os.getenv("QWEN_API_KEY", LLM_SERVER_KEY)
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5-plus")
LLM_PATH = _path("LLM_PATH", "./models/Qwen2.5-7B-Instruct")
LLM_DEVICE = os.getenv("LLM_DEVICE", "npu:0")
LLM_PORT = int(os.getenv("LLM_PORT", "8000"))

EVAL_BASE_URL = os.getenv("GJ_URL", BASE_URL)
EVAL_API_KEY = os.getenv("GJ_API_KEY", LLM_CLIENT_KEY)

DATA_DIR = _path("DATA_DIR", "./data/chunk_samples")
EVAL_DIR = _path("EVAL_DIR", "./data/eval_uploads")
KB1_DIR = _path("INDEX_DIR", "./basic_faiss_db")
KB2_DIR = _path("KB2_INDEX_DIR", "./basic_faiss_db2")
TEXT_EMBED_PATH = _path("TEXT_EMBED_PATH", "./embedding_model")

CHUNK_METHOD = os.getenv("CHUNK_METHOD", "parent_child")
METADATA_ENHANCE = _bool("METADATA_ENHANCE", True)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
HYBRID_FAISS_WEIGHT = float(os.getenv("HYBRID_FAISS_WEIGHT", "0.6"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
RERANK_PATH = _path("RERANK_PATH", "./models/bge-reranker-large")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
ENABLE_HYDE = _bool("ENABLE_HYDE", True)

ROUTER_DEVICE = os.getenv("ROUTER_DEVICE", "npu:0")
ROUTER_BASE_MODEL = _path("ROUTER_BASE_MODEL", "./bert-base-chinese")
ROUTER_MODEL_PATH = _path("ROUTER_MODEL_PATH", "./models/rag_switch")
FT_DATA = _path("FT_DATA", "./data/ft_data.csv")

WEB_PORT = int(os.getenv("WEB_PORT", "7861"))
API_PORT = int(os.getenv("API_PORT", "7862"))
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")
SERVICE_API_KEY = os.getenv("RAG_API_KEY", "")
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "10"))
HISTORY_REWRITE_ROUNDS = int(os.getenv("HISTORY_REWRITE_ROUNDS", "3"))
