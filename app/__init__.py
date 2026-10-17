"""RAG 应用包：昇腾 NPU 全本地化的私有知识库问答系统。

分层结构：
- core        通用基础设施（进度输出、LRU 缓存、OpenAI 兼容客户端）
- data        数据层（文档加载、文本分块、向量库与知识库管理）
- retrieval   检索层（混合检索、HyDE、重排、置信度过滤）
- generation  生成层（查询改写、答案反思、RAG 主编排）
- routing     路由层（BERT 意图 / 知识库路由）
- services    服务层（Flask 页面、FastAPI 接口、本地 LLM 服务）
- evaluation  评估层（RAGAS 在线 / 离线评估）
"""
