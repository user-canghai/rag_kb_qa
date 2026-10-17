"""BERT 意图路由：判断问题是否需要 RAG，以及使用哪个知识库。"""
import importlib
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

from app import config
from app.core.progress import banner, fmt_seconds, info, ok, section, track

warnings.filterwarnings("ignore")
importlib.import_module("torch_npu")

DEVICE = torch.device(config.ROUTER_DEVICE)
BASE_MODEL = config.ROUTER_BASE_MODEL
FT_DATA = config.FT_DATA
MODEL_PATH = config.ROUTER_MODEL_PATH
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
TEST_SIZE = 0.1

label2kb = {0: "kb1", 1: None, 2: "kb2"}

os.makedirs(MODEL_PATH, exist_ok=True)
torch.npu.set_device(DEVICE)

ROUTER_CACHE = {"tokenizer": None, "model": None}


class RAGRouterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_router_model():
    section(f"训练路由模型（设备 {DEVICE}，{EPOCHS} 个 epoch，batch {BATCH_SIZE}）")
    started = time.time()
    df = pd.read_csv(FT_DATA, header=None, names=["text", "label"])
    info(f"读取训练数据：{FT_DATA}，共 {len(df)} 条")
    train_text, val_text, train_label, val_label = train_test_split(
        df["text"].values, df["label"].values, test_size=TEST_SIZE, random_state=42
    )
    info(f"划分完成：训练 {len(train_text)} 条 / 验证 {len(val_text)} 条")

    tokenizer = BertTokenizer.from_pretrained(str(BASE_MODEL))
    model = BertForSequenceClassification.from_pretrained(
        str(BASE_MODEL), num_labels=len(label2kb), low_cpu_mem_usage=False
    ).to(DEVICE)

    train_loader = DataLoader(
        RAGRouterDataset(train_text, train_label, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        RAGRouterDataset(val_text, val_label, tokenizer), batch_size=BATCH_SIZE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        total_loss = 0.0
        for batch in track(
            train_loader,
            total=len(train_loader),
            desc=f"第 {epoch + 1}/{EPOCHS} 轮训练",
            unit="batch",
        ):
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["label"].to(DEVICE),
            )
            total_loss += outputs.loss.item()
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ok(
            f"第 {epoch + 1}/{EPOCHS} 轮完成：平均 loss "
            f"{total_loss / len(train_loader):.4f}，用时 {fmt_seconds(time.time() - epoch_started)}"
        )

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in track(val_loader, total=len(val_loader), desc="验证集评估", unit="batch"):
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            correct += (logits.argmax(1).cpu() == batch["label"]).sum().item()
    ok(f"验证集准确率：{correct / len(val_text):.4f}")

    clear_router_model_cache()
    info(f"保存模型到 {MODEL_PATH}")
    model.save_pretrained(
        str(MODEL_PATH), state_dict={k: v.cpu() for k, v in model.state_dict().items()}
    )
    tokenizer.save_pretrained(str(MODEL_PATH))
    ok(f"路由模型训练完成，总用时 {fmt_seconds(time.time() - started)}")


def get_router_model():
    if ROUTER_CACHE["model"] is None:
        info(f"首次加载路由模型：{MODEL_PATH}（之后直接走缓存）")
        started = time.time()
        tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))
        model = BertForSequenceClassification.from_pretrained(
            str(MODEL_PATH), low_cpu_mem_usage=False
        ).to(DEVICE)
        model.eval()
        ok(f"路由模型就绪，用时 {fmt_seconds(time.time() - started)}")
        ROUTER_CACHE["tokenizer"] = tokenizer
        ROUTER_CACHE["model"] = model
    return ROUTER_CACHE["tokenizer"], ROUTER_CACHE["model"]


def clear_router_model_cache():
    ROUTER_CACHE["tokenizer"] = None
    ROUTER_CACHE["model"] = None
    info("路由模型缓存已清空")


def vector_db_route(query: str) -> dict:
    if not (MODEL_PATH / "config.json").exists():
        info(f"未找到路由模型（{MODEL_PATH}），先自动训练一次")
        train_router_model()

    started = time.time()
    tokenizer, model = get_router_model()
    inputs = tokenizer(
        query, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(1).item()
        conf = float(torch.softmax(logits, dim=1).max().item())

    target_kb = label2kb.get(pred)
    info(
        f"多知识库路由：{query[:24]} -> label={pred}"
        f" · {'专业知识' if target_kb else '通用问题'}"
        f" · {target_kb if target_kb else '不检索'}"
        f"（置信度 {conf:.3f}，用时 {fmt_seconds(time.time() - started)}）"
    )
    return {
        "need_rag": target_kb is not None,
        "vector_db": target_kb,
        "query_type": "专业知识" if target_kb else "通用问题",
        "label": pred,
        "confidence": round(conf, 3),
    }


def need_rag(query: str) -> bool:
    return vector_db_route(query)["need_rag"]


if __name__ == "__main__":
    banner("rag_router.py - 多知识库路由自测", f"模型 {MODEL_PATH}，设备 {DEVICE}")
    test_questions = [
        "IPRAN专线网络告警分为哪三大类？",
        "你平时喜欢吃什么水果？",
        "IT运维工单规范适用于哪些人员？",
        "服务器宕机如何排查？",
        "数据备份策略是什么？",
    ]
    for q in test_questions:
        res = vector_db_route(q)
        print(
            f"问题：{q}\n标签：{res['label']} | 类型：{res['query_type']} | "
            f"库：{res['vector_db']} | 置信度：{res['confidence']}"
        )
