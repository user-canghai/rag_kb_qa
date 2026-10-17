#!/bin/bash
cd "$(dirname "$0")/.."
echo "===== 停止 RAG 服务 ====="
for name in llm_server rag; do
    if [ -f logs/$name.pid ]; then
        pid="$(cat logs/$name.pid)"
        if kill "$pid" 2>/dev/null; then
            echo "已停止 $name（PID $pid）"
        else
            echo "$name（PID $pid）未在运行"
        fi
        rm -f logs/$name.pid
    else
        echo "$name 没有 PID 文件，跳过"
    fi
done
