"""文本分块策略与元数据增强（5 种策略）。"""
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.progress import fmt_seconds, info, track


def fixed_size_split(text, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", " "],
    )
    return splitter.create_documents([text])


def sentence_split(text):
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r"(?<=[。！？；])", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    return [Document(page_content=s) for s in sentences]


def semantic_split(text, embed_model, threshold=75):
    from langchain_experimental.text_splitter import SemanticChunker

    sentence_docs = sentence_split(text)
    sentences = [doc.page_content for doc in sentence_docs]
    if len(sentences) <= 1:
        return sentence_docs
    chunker = SemanticChunker(
        embeddings=embed_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )
    try:
        chunks = chunker.create_documents(sentences)
        info(f"语义分块：{len(sentences)} 句 -> {len(chunks)} 块（阈值 {threshold} 分位）")
        return chunks
    except Exception as e:
        print(f"语义分块失败，使用句子分块: {e}")
        return sentence_docs


def parent_child_split(text, parent_size=1024, child_size=128):
    parent_docs = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=128
    ).create_documents([text])
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=16)
    child_docs = []
    for parent in parent_docs:
        for child in child_splitter.split_text(parent.page_content):
            child_docs.append(
                Document(page_content=child, metadata={"parent_content": parent.page_content})
            )
    return child_docs


def summary_lead_split(text, client, chunk_size=512):
    chunks = fixed_size_split(text, chunk_size, 64)
    info(f"摘要前置分块：{len(chunks)} 个块，每块都要调用一次大模型，耗时较长")
    enhanced = []
    started = time.time()
    for doc in track(chunks, total=len(chunks), desc="生成摘要", unit="块"):
        content = doc.page_content.strip()
        if not content:
            continue
        try:
            response = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[{"role": "user", "content": f"请用10字以内简洁概括这段文字：{content}"}],
                stream=False,
            )
            summary = response.choices[0].message.content.strip()
            content = f"【摘要】{summary}\n{content}"
        except Exception as e:
            print(f"摘要前置失败，保留原文: {e}")
        enhanced.append(Document(page_content=content))
    info(f"摘要前置完成：{len(enhanced)} 个块，用时 {fmt_seconds(time.time() - started)}")
    return enhanced


def add_metadata_enhance(docs, filename, chunk_method):
    for i, doc in enumerate(docs):
        doc.metadata.update(
            {
                "filename": filename,
                "chunk_index": i,
                "chunk_method": chunk_method,
                "length": len(doc.page_content),
            }
        )
    return docs
