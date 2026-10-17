"""RAGAS 在线批量评估：网页上传 JSON 数据集后在后台线程评估。"""
import json
import os
import time

from app import config
from app.core.progress import fmt_seconds, info, ok, section
from app.state import state

os.environ["OPENAI_API_KEY"] = config.EVAL_API_KEY
os.environ["OPENAI_BASE_URL"] = config.EVAL_BASE_URL


def run_ragas_eval(eval_json_path: str) -> None:
    from datasets import Dataset
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    from app.data.embeddings import ensure_embed_model

    section("RAGAS 批量评估")
    info(f"评估文件：{eval_json_path}")

    progress = state.ragas_progress
    try:
        state.ragas_result.clear()
        progress.update({"status": "评估中", "progress": 0, "current": 0, "total": 0})

        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        total_qa = len(eval_data)
        ok(f"读取到 {total_qa} 条评估数据，即将评估 3 项指标（较慢，请等待）")
        progress["total"] = total_qa * 3

        questions, answers, contexts, ground_truths = [], [], [], []
        for item in eval_data:
            questions.append(item["query"])
            answers.append(item["answer"])
            ctx = item["document"]
            contexts.append(ctx if isinstance(ctx, list) else [ctx])
            ground_truths.append(item["answer"])

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        def adapt_messages_for_qwen(llm):
            """兼容补丁：本地模型链路对 system 消息不友好，统一下降级为 human 消息。"""
            original_generate = llm._generate

            def patched_generate(messages, stop=None, **kwargs):
                adapted = []
                for msg in messages:
                    content = msg.content
                    if not isinstance(content, str):
                        content = str(content)
                    if isinstance(msg, SystemMessage):
                        adapted.append(HumanMessage(content=f"系统：{content}"))
                    else:
                        adapted.append(msg.__class__(content=content))
                return original_generate(adapted, stop=stop, **kwargs)

            llm._generate = patched_generate
            return llm

        llm = ChatOpenAI(
            model=config.MODEL_NAME,
            openai_api_key=config.EVAL_API_KEY,
            base_url=config.EVAL_BASE_URL,
            temperature=0.1,
            model_kwargs={"extra_body": {"enable_thinking": False}},
        )
        llm = adapt_messages_for_qwen(llm)

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeds = LangchainEmbeddingsWrapper(ensure_embed_model())

        m1 = Faithfulness(llm=ragas_llm)
        m2 = AnswerRelevancy(llm=ragas_llm)
        m3 = ContextPrecision(llm=ragas_llm)

        info("正在评估：检索相关性 / 生成与检索相关性 / 生成回答相关性")
        started = time.time()
        res = evaluate(
            dataset=dataset, llm=ragas_llm, embeddings=ragas_embeds, metrics=[m1, m2, m3]
        )

        def safe(v):
            if isinstance(v, list):
                v = sum(v) / len(v) if len(v) else 0.0
            return round(float(v), 2) if v is not None else 0.0

        state.ragas_result.update(
            {
                "检索相关性": safe(res["context_precision"]),
                "生成与检索相关性": safe(res["faithfulness"]),
                "生成回答相关性": safe(res["answer_relevancy"]),
            }
        )
        progress.update({"status": "评估完成", "current": 3 * total_qa, "progress": 100})
        ok(f"评估完成，用时 {fmt_seconds(time.time() - started)}")
        ok(f"最终结果：{state.ragas_result}")

    except Exception as e:
        state.ragas_result.clear()
        state.ragas_result["评估失败"] = str(e)
        progress["status"] = "评估失败"
        print("评估失败：", str(e))
