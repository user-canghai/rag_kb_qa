"""Flask 页面服务：RAG 问答界面 + 文件管理 + RAGAS 触发。

路由一览：
- GET  /                     首页（文件列表、对话历史、评估结果）
- POST /stream_chat          流式问答（auto / rag / no_rag 三种模式）
- POST /upload               上传文档并即时入库（固定写知识库1）
- POST /delete_selected      删除选中文档与对应索引
- POST /build_selected_index 重建选中文档的索引
- POST /upload_eval          RAGAS 批量评估（后台线程执行）
- GET  /clear_history        清空对话历史与问答缓存
- GET  /clear_all            清空文档 + 两个知识库 + 评估上传 + 缓存（脚本可复原）
- GET  /eval_progress        评估进度轮询接口
"""
import os
import shutil
import time
from pathlib import Path
from threading import Thread

from flask import Flask, Response, jsonify, redirect, render_template, request

from app import config
from app.core.progress import banner, fmt_seconds, info, kv, section, track
from app.data import vector_store
from app.evaluation import ragas_batch
from app.generation import pipeline
from app.state import state

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = config.FLASK_SECRET_KEY or os.urandom(32)

for _d in (config.DATA_DIR, config.EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def clear_all_cache():
    """清空对话与问答缓存；路由模型若已加载则一并释放。"""
    state.clear_conversation()
    print("问答缓存已清空")
    import sys

    router_module = sys.modules.get("app.routing.router")
    if router_module is not None:
        router_module.clear_router_model_cache()


def resolve_mode(query, mode):
    """解析请求模式，返回 (生效模式, 目标知识库目录或 None)。"""
    if mode == "auto":
        from app.routing import router

        route = router.vector_db_route(query)
        if route["need_rag"]:
            kb_dir = config.KB1_DIR if route["vector_db"] == "kb1" else config.KB2_DIR
            return "rag", kb_dir
        return "no_rag", None
    if mode == "rag":
        return "rag", config.KB1_DIR
    return "no_rag", None


@app.route("/")
def index():
    state.chat_history = []
    file_list = sorted(os.listdir(config.DATA_DIR)) if config.DATA_DIR.exists() else []
    return render_template(
        "index.html", history=state.chat_history, files=file_list, ragas=state.ragas_result
    )


@app.route("/stream_chat", methods=["POST"])
def stream_chat():
    started = time.time()
    data = request.get_json()
    query = data.get("query", "")
    mode = data.get("mode", "rag")

    mode, kb_dir = resolve_mode(query, mode)
    cache_key = (mode, str(kb_dir), query)
    cached_answer = state.qa_cache.get(cache_key)
    if cached_answer is not None:
        info(f"命中问答缓存，直接返回（等待 {fmt_seconds(time.time() - started)}）")
        return Response(iter([cached_answer]), mimetype="text/plain")

    if mode == "rag":
        vector_store.use_kb(kb_dir)
    gen_func = pipeline.generate_rag if mode == "rag" else pipeline.generate_plain

    def stream_and_log():
        for chunk in pipeline.stream_and_cache(query, mode, cache_key, gen_func):
            yield chunk
        info(f"本次请求处理完毕，总耗时 {fmt_seconds(time.time() - started)}")

    return Response(stream_and_log(), mimetype="text/plain")


@app.route("/upload", methods=["POST"])
def upload():
    uploads = [f for f in request.files.getlist("file") if f.filename]
    section(f"上传文档（{len(uploads)} 个）")
    vector_store.use_kb1()
    for f in track(uploads, total=len(uploads), desc="上传并建索引", unit="个"):
        path = config.DATA_DIR / os.path.basename(f.filename)
        info(f"保存文件：{path.name}")
        f.save(str(path))
        vector_store.add_file_to_index(path)
    clear_all_cache()
    return redirect("/")


@app.route("/delete_selected", methods=["POST"])
def delete_selected():
    selected = request.form.getlist("selected_files")
    section(f"删除文档（{len(selected)} 个）")
    vector_store.use_kb1()
    for f in track(selected, total=len(selected), desc="删除文档", unit="个"):
        p = config.DATA_DIR / f
        info(f"删除文件：{f}")
        if p.exists():
            p.unlink()
        vector_store.delete_file_from_index(f)
    clear_all_cache()
    return redirect("/")


@app.route("/build_selected_index", methods=["POST"])
def build_selected_index():
    selected = request.form.getlist("selected_files")
    section(f"重建选中文档的索引（{len(selected)} 个）")
    vector_store.use_kb1()
    for f in track(selected, total=len(selected), desc="重建索引", unit="个"):
        p = config.DATA_DIR / f
        if p.exists():
            info(f"先删除旧索引，再重新入库：{f}")
            vector_store.delete_file_from_index(f)
            vector_store.add_file_to_index(p)
    clear_all_cache()
    return redirect("/")


@app.route("/upload_eval", methods=["POST"])
def upload_eval():
    try:
        state.reset_ragas()
        section("收到 RAGAS 批量评估请求")

        f = request.files.get("eval_file")
        if not f:
            print("【错误】未获取到文件！")
            return redirect("/")

        save_path = config.EVAL_DIR / os.path.basename(f.filename)
        f.save(str(save_path))
        print(f"【成功】文件已保存：{save_path}")

        Thread(target=ragas_batch.run_ragas_eval, args=(str(save_path),), daemon=True).start()
        return redirect("/")
    except Exception as e:
        print("上传异常：", e)
        return redirect("/")


@app.route("/clear_history")
def clear_history():
    section("清空对话历史")
    state.clear_conversation()
    print("问答缓存已清空")
    return redirect("/")


@app.route("/clear_all")
def clear_all():
    """清空全部数据。"""
    section("清空全部数据（文档 + 两个知识库索引 + 评估上传 + 缓存）")
    with state.lock:
        state.clear_conversation()
        state.reset_ragas()
        state.index_dir = None
        state.reset_database()
        for kb_dir in (config.KB1_DIR, config.KB2_DIR):
            shutil.rmtree(kb_dir, ignore_errors=True)
            kb_dir.mkdir(parents=True, exist_ok=True)
        for data_dir in (config.DATA_DIR, config.EVAL_DIR):
            shutil.rmtree(data_dir, ignore_errors=True)
            data_dir.mkdir(parents=True, exist_ok=True)
    clear_all_cache()
    return redirect("/")


@app.route("/eval_progress")
def eval_progress():
    return jsonify(
        {
            "progress": state.ragas_progress["progress"],
            "status": state.ragas_progress["status"],
            "current": state.ragas_progress["current"],
            "total": state.ragas_progress["total"],
            "result": state.ragas_result,
        }
    )


def main():
    banner(
        "RAG 私有知识库问答系统 - Web 服务",
        "启动后浏览器访问 http://<服务器IP>:7861 ，终端会实时打印每一步处理过程",
    )
    kv(
        [
            ("大模型服务", f"{config.BASE_URL}（模型名 {config.MODEL_NAME}）"),
            ("知识库1", str(config.KB1_DIR)),
            ("知识库2", str(config.KB2_DIR)),
            ("分块方式", f"{config.CHUNK_METHOD}（元数据增强 {config.METADATA_ENHANCE}）"),
            ("重排模型", str(config.RERANK_PATH)),
            ("路由设备", config.ROUTER_DEVICE),
        ]
    )
    section("启动 Flask 服务")
    app.run(host="0.0.0.0", port=config.WEB_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
