import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness
)
from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
import torch
from app import config
from app.data.embeddings import FastEmbedEmbeddings
from app.data.splitters import (
    fixed_size_split,
    parent_child_split,
    semantic_split,
    sentence_split,
    summary_lead_split,
)
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

API_KEY = config.EVAL_API_KEY
BASE_URL = config.EVAL_BASE_URL

os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


class ChunkingStrategyEvaluator:
    def __init__(self, embedding_model, llm_model):
        self.embeddings = embedding_model
        self.llm = llm_model
        self.evaluation_results = {}        
        self.use_faiss = True  

    def evaluate_strategy(self, strategy_name: str, chunking_func, documents: list[str]):
        """评估单个分块策略"""
        print(f"\n正在评估策略: {strategy_name}")
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = chunking_func(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  分块失败: {e}")
                continue
        
        print(f"  生成 {len(all_chunks)} 个块")
        
        if len(all_chunks) == 0:
            print("  未生成任何块，跳过")
            return None
        
        from langchain_community.vectorstores import FAISS        
        vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        def rag_query(question: str) -> Dict[str, Any]:
            try:
                docs = retriever.invoke(question)
                if strategy_name == "parent_child":
                    context = "\n".join(list(set([doc.metadata["parent_content"] for doc in docs])))
                else:
                    context = "\n".join([doc.page_content for doc in docs])
                
                prompt = f"""基于以下上下文回答问题。如果无法从上下文中找到答案，请明确说明。                
                        上下文：
                        {context}
                        问题：{question}
                        答案："""
                
                response = self.llm.invoke(prompt)
                
                return {
                    "answer": response.content,
                    "contexts": [doc.page_content for doc in docs]
                }
            except Exception as e:
                print(f"  查询失败: {e}")
                return {
                    "answer": f"生成失败: {e}",
                    "contexts": []
                }
        
        return rag_query
    
    def cleanup(self):
        """无需清理 FAISS"""
        pass


def run_ragas_evaluation(test_dataset, strategy_name, rag_func, llm_model,embed_model):
    """使用RAGAS框架进行评估"""
    
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for test_case in tqdm(test_dataset, desc=f"评估 {strategy_name}"):
        try:
            result = rag_func(test_case["question"])
            eval_data["question"].append(test_case["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append(result["contexts"])
            eval_data["ground_truth"].append(test_case["ground_truth"])
        except Exception as e:
            print(f"  问题处理失败: {e}")
            continue
    
    if len(eval_data["question"]) == 0:
        print("  没有有效的问题结果")
        return None
    
    dataset = Dataset.from_dict(eval_data)   
    
    try:
        
        result = evaluate(
            dataset=dataset,
            embeddings=embed_model,
            metrics=[
                ContextRecall(llm=llm_model),
                ContextPrecision(llm=llm_model),
                AnswerRelevancy(llm=llm_model),
                Faithfulness(llm=llm_model)
            ]
            
        )
        return result
    except Exception as e:
        print(f"  RAGAS 评估失败: {e}")
        return None


def compare_all_strategies(test_documents, test_dataset, embedding_model, llm_model, llm_client):
    """对比所有分块策略"""
    
    strategies = {      
        "fixed_size": lambda text: fixed_size_split(text, chunk_size=256, chunk_overlap=64),        
        "sentence_split": sentence_split,
        "parent_child": lambda text: parent_child_split(text,parent_size=1024, child_size=256),
        "summary_prefix": lambda text: summary_lead_split(text, llm_client,chunk_size=256),
        "semantic_basic": lambda text: semantic_split(text, embedding_model, threshold=75),
    }
    
    evaluator = ChunkingStrategyEvaluator(embedding_model, llm_model)
    results = {}
    
    try:
        for strategy_name, chunk_func in tqdm(strategies.items(), desc="评估所有策略"):
            print(f"\n{'='*50}")
            print(f"正在评估: {strategy_name}")
            print(f"{'='*50}")
            
            try:
                rag_func = evaluator.evaluate_strategy(strategy_name, chunk_func, test_documents)
                
                if rag_func is None:
                    print(f"✗ 策略 {strategy_name} 初始化失败")
                    results[strategy_name] = None
                    continue
                
                eval_result = run_ragas_evaluation(
                    test_dataset, 
                    strategy_name, 
                    rag_func,
                    llm_model,
                    embedding_model
                )
                
                if eval_result:
                    results[strategy_name] = eval_result
                    print(f"\n✓ {strategy_name} 评估完成:")
                    print(f"  - Context Recall: {np.mean(eval_result['context_recall'])}")
                    print(f"  - Context Precision: {np.mean(eval_result['context_precision'])}")
                    print(f"  - Answer Relevancy: {np.mean(eval_result['answer_relevancy'])}")
                    print(f"  - Faithfulness: {np.mean(eval_result['faithfulness'])}")

                else:
                    results[strategy_name] = None
                    
            except Exception as e:
                print(f"✗ 策略 {strategy_name} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                results[strategy_name] = None
    finally:
        evaluator.cleanup()  
    
    return results


def analyze_results(results: Dict[str, Any]):
    """分析评估结果"""
       
    data = []
    for name, r in results.items():
        print(r)
        cr = np.mean(r['context_recall'])
        cp = np.mean(r['context_precision'])
        ar = np.mean(r['answer_relevancy'])
        ft = np.mean(r['faithfulness'])
        data.append({
            "策略": name,
            "召回": round(cr,2),
            "精度": round(cp,2),
            "相关性": round(ar,2),
            "忠实度": round(ft,2),
            "综合": round((cr+cp+ar+ft)/4, 2)
        })

    df = pd.DataFrame(data)
    df = df.sort_values("综合", ascending=False)
    print("\n 最终排名（从高到低）")
    print(df.to_string(index=False))
    return df.iloc[0]["策略"], df


def setup_llm_clients():
    """设置LLM客户端"""   
    
    torch.set_num_threads(8)
    
    embedding_model = FastEmbedEmbeddings()
    
    llm_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )    
    
    llm_model = ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        n=1,
        temperature=0.1,        
        stream_usage=False
    )
    
    return embedding_model, llm_model, llm_client


def load_documents_from_files(data_dir_path):
    """从文件加载文档"""
    file_paths = os.listdir(data_dir_path)
    documents = []
    for file_path in file_paths:
        try:
            with open(os.path.join(data_dir_path,file_path), 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
            print(f"✓ 已加载文档: {file_path}")
        except Exception as e:
            print(f"✗ 加载失败 {file_path}: {e}")
    return documents

def get_test_questions():
    """【长文本专用】RAG分块策略测试问题集（完全匹配4篇2000字文档）"""
    test_questions = [
        {
            "question": "当 IPRAN 网络出现 81a4 主用链路信号丢失告警，且经过 ODF 检查、割接确认、光纤损耗测试后仍未恢复，此时应该执行什么操作？完整操作结束后还需要完成哪些后续工作？",
            "ground_truth": "应立即执行备用链路切换，将业务切换至备用链路以恢复业务；故障恢复后需要记录故障信息、生成闭环工单、完成故障复盘并归档至运维知识库。",
            
        }        
    ]

    return test_questions


if __name__ == "__main__":
    print("="*80)
    print("RAG系统分块策略对比评估系统")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("使用示例文档进行测试")

    data_dir_path = "./data/chunk_samples"    
    test_documents = load_documents_from_files(data_dir_path)
    print(f"\n✓ 已加载 {len(test_documents)} 个文档")
    print(f"  文档总长度: {sum(len(doc) for doc in test_documents)} 字符")

    test_questions = get_test_questions()
    print(f"\n✓ 已准备 {len(test_questions)} 个测试问题")

    print("\n正在初始化LLM客户端...")
    embedding_model, llm_model, llm_client = setup_llm_clients()
    print("✓ LLM客户端初始化完成")

    print("\n开始评估分块策略...")
    print("="*80)
    
    results = compare_all_strategies(
        test_documents=test_documents,
        test_dataset=test_questions,
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_client=llm_client
    )
    
    best_strategy, analysis_df = analyze_results(results)
    
    if analysis_df is not None:
        os.makedirs("results", exist_ok=True)
        report = os.path.join("results", f"chunking_analysis_{timestamp}.csv")
        analysis_df.to_csv(report, index=False)
        print(f"\n✓ 分析报告已保存: {report}")
