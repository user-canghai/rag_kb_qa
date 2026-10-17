#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs
source .venv/bin/activate
source scripts/env.sh

echo "===== 启动 RAG 服务 ====="
echo "项目目录：$(pwd)"
echo "日志目录：$(pwd)/logs"
echo

echo "[1/2] 检查本地大模型服务（127.0.0.1:8000）..."
if ! curl -s -m 3 http://127.0.0.1:8000/health > /dev/null; then
    echo "      未运行，后台启动 app.services.llm_server"
    nohup python -m app.services.llm_server > logs/llm_server.log 2>&1 &
    echo $! > logs/llm_server.pid
    echo "      已启动（PID $(cat logs/llm_server.pid)）"
else
    echo "      已在运行，跳过"
fi

echo "[2/2] 检查 RAG Web 服务（127.0.0.1:7861）..."
if ! curl -s -m 3 http://127.0.0.1:7861/ > /dev/null; then
    echo "      未运行，后台启动 app.services.web"
    nohup python -m app.services.web > logs/rag.log 2>&1 &
    echo $! > logs/rag.pid
    echo "      已启动（PID $(cat logs/rag.pid)）"
else
    echo "      已在运行，跳过"
fi

echo
echo "===== 启动完成 ====="
echo "查看实时处理过程："
echo "  tail -f logs/rag.log          每次提问的分步进度与耗时"
echo "  tail -f logs/llm_server.log   大模型加载与生成速率"
echo
echo "想在终端直接看输出（推荐）：先 ./scripts/stop.sh，再前台运行 python -m app.services.web"
