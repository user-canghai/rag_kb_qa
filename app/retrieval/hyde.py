"""HyDE 假设文档增强：先让大模型生成假设答案，再用它去检索。"""
import time

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app import config
from app.core.progress import fmt_seconds, info

HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是专业IT运维、IPRAN、服务器技术专家。
根据问题生成一段专业、严谨、符合知识库风格的模拟答案，要求500字内。
问题：{question}
模拟答案：""",
)


def generate_hyde_query(query: str) -> str:
    """生成 HyDE 假设答案；失败时回退原问题。"""
    started = time.time()
    try:
        llm = ChatOpenAI(
            model=config.MODEL_NAME,
            openai_api_base=config.BASE_URL,
            openai_api_key=config.LLM_CLIENT_KEY,
            temperature=0.0,
        )
        hyde_text = (HYDE_PROMPT | llm).invoke({"question": query}).content.strip()
        info(f"HyDE 生成假设答案 {len(hyde_text)} 字，用时 {fmt_seconds(time.time() - started)}")
        return hyde_text
    except Exception as e:
        print(f"HyDE 失败，使用原问题：{e}")
        return query
