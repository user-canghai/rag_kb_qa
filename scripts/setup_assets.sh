#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh

BERT_DIR=bert-base-chinese
if [ ! -f "$BERT_DIR/config.json" ]; then
    echo "下载 bert-base-chinese ..."
    mkdir -p "$BERT_DIR"
    BASE="https://modelscope.cn/models/google-bert/bert-base-chinese/resolve/master"
    for f in config.json vocab.txt tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors; do
        curl -sSL -o "$BERT_DIR/$f" "$BASE/$f" &
    done
    wait
fi

if [ ! -f models/rag_switch/config.json ]; then
    echo "训练 BERT 路由模型 ..."
    python -m app.routing.router | tail -4
fi

if [ ! -d embedding_model/fast-bge-small-zh-v1.5 ]; then
    echo "缺少 embedding_model/，请按 README「环境要求」准备"
fi

echo "模型资产就绪"
