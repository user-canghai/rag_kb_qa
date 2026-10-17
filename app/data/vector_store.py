"""向量库与知识库管理：Embedding / FAISS 生命周期、双库切换、入库与删除。"""
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app import config
from app.core.llm import get_client
from app.core.progress import fmt_seconds, info, ok, track
from app.data.embeddings import ensure_embed_model
from app.data.loaders import load_single_file
from app.data.splitters import (
    add_metadata_enhance,
    fixed_size_split,
    parent_child_split,
    semantic_split,
    sentence_split,
    summary_lead_split,
)
from app.state import state

INDEX_FILE_FAISS = "index.faiss"
INDEX_FILE_PKL = "index.pkl"


def current_kb() -> Path:
    """当前知识库目录（未切换过则为默认库 KB1）。"""
    return state.index_dir or config.KB1_DIR


def use_kb(kb_dir) -> None:
    """切换当前知识库；索引与 BM25 缓存下次使用前重建。"""
    kb_dir = Path(kb_dir)
    with state.lock:
        if state.index_dir != kb_dir:
            state.index_dir = kb_dir
            state.reset_database()
            info(f"切换知识库：{kb_dir}")


def use_kb1() -> None:
    """文件面板固定管理知识库1：切换 + 确保模型与索引就绪。"""
    use_kb(config.KB1_DIR)
    ensure_embed_model()
    ensure_db()


def reset_database() -> None:
    """清空内存中的索引（下次 ensure_db() 重新加载）。"""
    with state.lock:
        state.reset_database()


def _index_files_exist(kb_dir: Path) -> bool:
    return (kb_dir / INDEX_FILE_FAISS).exists() and (kb_dir / INDEX_FILE_PKL).exists()


def ensure_db() -> FAISS | None:
    """确保当前知识库可用：有索引则加载；默认库缺索引时从 DATA_DIR 现场构建。"""
    if state.faiss_db is not None:
        return state.faiss_db
    embed_model = ensure_embed_model()
    kb_dir = current_kb()
    if _index_files_exist(kb_dir):
        info(f"加载向量库：{kb_dir}")
        started = time.time()
        state.faiss_db = FAISS.load_local(
            str(kb_dir), embed_model, allow_dangerous_deserialization=True
        )
        ok(
            f"向量库就绪：{state.faiss_db.index.ntotal} 个向量块，"
            f"用时 {fmt_seconds(time.time() - started)}"
        )
    elif kb_dir == config.KB1_DIR and config.DATA_DIR.exists():
        info(f"未找到索引文件，改为从 {config.DATA_DIR} 现场构建")
        build_index(config.DATA_DIR, config.KB1_DIR)
    else:
        info(f"知识库为空：{kb_dir}")
    return state.faiss_db


def split_document(text: str):
    """按 config.CHUNK_METHOD 选择分块策略。"""
    if config.CHUNK_METHOD == "sentence":
        chunks = sentence_split(text)
    elif config.CHUNK_METHOD == "semantic":
        chunks = semantic_split(text, ensure_embed_model())
    elif config.CHUNK_METHOD == "parent_child":
        chunks = parent_child_split(text, parent_size=1024, child_size=256)
    elif config.CHUNK_METHOD == "summary_lead":
        chunks = summary_lead_split(text, get_client(), chunk_size=config.CHUNK_SIZE)
    else:
        chunks = fixed_size_split(text, config.CHUNK_SIZE, 64)
    info(f"分块完成：策略 {config.CHUNK_METHOD}，{len(text)} 字 -> {len(chunks)} 块")
    return chunks


def add_file_to_index(file_path, save: bool = True) -> None:
    """单文件入库（增量）；save=False 供批量构建时最后统一落盘。"""
    started = time.time()
    with state.lock:
        ensure_embed_model()
        doc = load_single_file(file_path)
        if not doc:
            return
        chunks = split_document(doc.page_content)
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        if config.METADATA_ENHANCE:
            chunks = add_metadata_enhance(chunks, doc.metadata["file"], config.CHUNK_METHOD)
            info(f"元数据增强：为 {len(chunks)} 个文本块补充 filename/chunk_index/chunk_method/length")
        if state.faiss_db is None:
            info(f"新建向量库：正在为 {len(chunks)} 个文本块计算向量")
            state.faiss_db = FAISS.from_documents(chunks, state.embed_model)
        else:
            info(f"增量入库：正在为 {len(chunks)} 个文本块计算向量")
            state.faiss_db.add_documents(chunks)
        state.bm25 = None
        if save:
            state.faiss_db.save_local(str(current_kb()))
        ok(
            f"入库完成：{doc.metadata['file']} 新增 {len(chunks)} 块，"
            f"当前共 {state.faiss_db.index.ntotal} 块，用时 {fmt_seconds(time.time() - started)}"
        )


def delete_file_from_index(filename: str) -> None:
    """按文件名删除该文件的所有子块（同时匹配 file / filename）并落盘。"""
    with state.lock:
        if state.faiss_db is None:
            return
        ids = []
        for cid in state.faiss_db.index_to_docstore_id.values():
            metadata = state.faiss_db.docstore.search(cid).metadata
            if filename in (metadata.get("file"), metadata.get("filename")):
                ids.append(cid)
        info(f"删除索引：{filename} 命中 {len(ids)} 个文本块")
        if ids:
            state.faiss_db.delete(ids)
            state.bm25 = None
            state.faiss_db.save_local(str(current_kb()))
            ok(f"删除完成，向量库剩余 {state.faiss_db.index.ntotal} 个向量块")


def build_index(data_dir, index_dir) -> None:
    """全量重建：从 data_dir 读取全部文件写入 index_dir（命令行入口调用）。"""
    data_dir = Path(data_dir)
    with state.lock:
        use_kb(index_dir)
        state.reset_database()
        ensure_embed_model()
        file_names = sorted(p.name for p in data_dir.iterdir() if p.is_file())
        info(f"开始构建索引：{data_dir}，共 {len(file_names)} 个文件")
        for file_name in track(file_names, total=len(file_names), desc="构建索引", unit="个"):
            add_file_to_index(data_dir / file_name, save=False)
        if state.faiss_db is not None:
            state.faiss_db.save_local(str(Path(index_dir)))
            ok(f"索引已保存：{index_dir}，共 {state.faiss_db.index.ntotal} 个向量块")
