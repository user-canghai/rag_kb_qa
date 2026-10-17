#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
source scripts/env.sh
mkdir -p data/chunk_samples data/documents basic_faiss_db basic_faiss_db2
cp -f data/backup/chunk_samples/* data/chunk_samples/
cp -f data/backup/documents/* data/documents/
rm -rf basic_faiss_db/* basic_faiss_db2/*
python -m app.data.build_vector ./data/chunk_samples ./basic_faiss_db
python -m app.data.build_vector ./data/documents ./basic_faiss_db2
echo "知识库文档与向量库已恢复"
