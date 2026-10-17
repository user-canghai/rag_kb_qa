# rag-kb-qa

本地化 RAG 私有知识库问答系统，覆盖文档解析、文本分块、向量入库、混合检索、重排序、置信度过滤、大模型生成、答案反思、意图路由与 RAGAS 评估。

## 功能特性

- 多格式文档加载：txt、pdf、docx、csv
- 五种分块策略：fixed、sentence、semantic、parent-child、summary-lead
- 混合检索：FAISS 向量检索与 BM25 关键词检索加权融合
- HyDE 假设文档增强、CrossEncoder 重排序、置信度过滤
- 多轮查询改写、答案自我反思
- BERT 三分类意图路由，支持双知识库切换
- 流式问答页面、SSE 服务接口、OpenAI 兼容本地模型服务
- RAGAS 在线批量评估与分块策略对比评估
- LRU 问答缓存与运行时状态集中管理

## 技术栈

- Python 3.11
- Flask、FastAPI、uvicorn
- LangChain、FAISS、BM25
- fastembed、sentence-transformers、transformers
- PyTorch、torch_npu（由运行环境提供）
- RAGAS、datasets

## 目录结构

```text
app/
  config.py              配置入口
  state.py               运行时状态
  core/                  进度输出、缓存、LLM 客户端
  data/                  文档加载、分块、向量库
  retrieval/             混合检索、HyDE、重排、置信度过滤
  generation/            查询改写、答案反思、RAG 编排
  routing/               BERT 意图路由
  services/              Web、API、本地 LLM 服务
  evaluation/            RAGAS 评估
data/
  backup/                原始示例文档备份
  chunk_samples/         知识库 1 文档
  documents/             知识库 2 文档
  ft_data.csv            路由模型训练数据
  test.pdf               PDF 加载测试样本
examples/                接口调用示例
scripts/                 启动、停止、数据恢复、模型资产脚本
.env.example             配置模板
README.md                项目说明
pyproject.toml           项目元数据与依赖
requirements.txt         运行依赖
requirements-dev.txt     开发依赖
uv.lock                  依赖锁定文件
```

## 环境要求

- Python 3.11
- Ascend NPU 与 CANN 运行环境（本地 LLM 服务、路由模型训练与自动路由需要）
- 本地模型资产（Embedding、重排序、路由底座与生成模型），路径通过 `.env` 配置

`torch_npu` 不写入依赖锁定文件；`torch` 会作为 `sentence-transformers` 的传递依赖被锁定。NPU 环境请保留预装的 `torch` 与 `torch_npu`。

## 快速开始

```bash
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# 按实际环境修改 .env

./scripts/setup_assets.sh
./scripts/reset_data.sh
./scripts/start.sh
```

首次运行或索引缺失时执行 `./scripts/reset_data.sh` 重建索引即可。

使用 uv 时，可在已激活的虚拟环境中安装依赖：

```bash
uv pip install -r requirements.txt
```

如需按 `uv.lock` 创建独立环境，可运行 `uv sync --no-install-package torch`。该命令不会安装 `torch`，需要另外安装与 CANN 匹配的 `torch` 与 `torch_npu`。

启动后访问 `http://127.0.0.1:7861`。

远程服务器可通过 SSH 隧道访问：

```bash
ssh -L 17861:127.0.0.1:7861 root@<服务器IP>
```

然后在浏览器打开 `http://127.0.0.1:17861`。

停止服务：

```bash
./scripts/stop.sh
```

## 常用入口

```bash
python -m app.services.llm_server
python -m app.services.web
python -m app.services.api
python -m app.data.build_vector <数据目录> <索引目录>
python -m app.routing.router
python -m app.evaluation.ragas_chunks
```

## 接口调用

先启动 API 服务：

```bash
API_PORT=7862 python -m app.services.api
```

再运行示例客户端：

```bash
python -m examples.client_request
```

接口需要 Bearer 鉴权，密钥由 `.env` 中的 `RAG_API_KEY` 配置。

## 代码检查

```bash
pip install -r requirements-dev.txt
ruff check --select F,E9 app examples
```

## 依赖锁定

项目使用 `uv.lock` 锁定依赖树。`torch_npu` 不在锁文件中；`torch` 会作为 `sentence-transformers` 的传递依赖被锁定。NPU 环境建议在 `--system-site-packages` 创建的虚拟环境中使用 `pip install -r requirements.txt` 或 `uv pip install -r requirements.txt`。若使用 `uv sync` 创建独立环境，请加 `--no-install-package torch`，并另外安装与 CANN 匹配的 `torch` 与 `torch_npu`。
