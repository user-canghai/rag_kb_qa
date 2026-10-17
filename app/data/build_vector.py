"""构建向量库命令行入口。"""
import sys
import time

from app.core.progress import banner, fmt_seconds, kv, ok
from app.data import vector_store
from app.state import state


def main() -> None:
    banner("build_vector - 构建向量库")
    started = time.time()
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/documents"
    index_dir = sys.argv[2] if len(sys.argv) > 2 else "./basic_faiss_db2"
    kv([("数据目录", data_dir), ("索引目录", index_dir)])
    vector_store.build_index(data_dir, index_dir)
    total = state.faiss_db.index.ntotal if state.faiss_db is not None else 0
    ok(
        f"向量库构建完成：{index_dir} 共 {total} 个向量块"
        f"（来自 {data_dir}），总用时 {fmt_seconds(time.time() - started)}"
    )


if __name__ == "__main__":
    main()
