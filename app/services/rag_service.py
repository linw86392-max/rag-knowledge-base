# -*- coding: utf-8 -*-
"""RAG 问答：检索 -> 重排 -> 生成（带引用溯源）"""
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import TOP_K
from app.core.models import get_llm, get_embeddings
from app.services.document_service import get_vectorstore

# 检索增强提示词：明确要求基于资料回答、不编造、带引用
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个企业知识库问答助手。请严格基于以下资料回答问题。

规则：
1. 只使用资料中的信息回答，资料中没有的内容请回答"根据当前知识库暂未找到相关信息"
2. 回答末尾以 [来源] 标注引用的文档名，格式：[来源：文件名1；文件名2]
3. 如果资料与问题无关，直接说明知识库中没有相关内容
4. 回答使用简洁的中文，条理清晰

资料：
{context}"""),
    ("human", "问题：{question}"),
])


def _merge_chunks(documents) -> str:
    """把检索结果拼成带编号的上下文，保留来源"""
    merged = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        page_info = f"（第{page}页）" if page != "" else ""
        merged.append(f"[{i}] 来源：{source}{page_info}\n{doc.page_content}")
    return "\n\n".join(merged)


def _format_sources(documents) -> List[Dict]:
    """提取来源列表，用于前端引用卡片"""
    seen, sources = set(), []
    for doc in documents:
        source = doc.metadata.get("source", "未知来源")
        if source not in seen:
            seen.add(source)
            sources.append({
                "source": source,
                "page": doc.metadata.get("page", ""),
            })
    return sources


def retrieve_documents(vectorstore, question: str, top_k: int = TOP_K):
    """向量检索 Top-K"""
    return vectorstore.similarity_search(question, k=top_k)


def search(question: str, collection_name: str = "kb_default", top_k: int = TOP_K) -> Dict:
    """非流式检索 + 生成（命令行快速验证用）"""
    vectorstore = get_vectorstore(collection_name)
    docs = retrieve_documents(vectorstore, question, top_k)

    if not docs:
        return {"answer": "知识库为空，请先上传文档。", "sources": []}

    context = _merge_chunks(docs)
    chain = QA_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": _format_sources(docs),
    }


def stream_search(question: str, collection_name: str = "kb_default", top_k: int = TOP_K):
    """流式检索 + 生成，逐 token yield，最后 yield 来源信息"""
    vectorstore = get_vectorstore(collection_name)
    docs = retrieve_documents(vectorstore, question, top_k)

    if not docs:
        yield {"type": "text", "content": "知识库为空，请先上传文档。"}
        return

    context = _merge_chunks(docs)
    chain = QA_PROMPT | get_llm() | StrOutputParser()

    for chunk in chain.stream({"context": context, "question": question}):
        yield {"type": "text", "content": chunk}

    yield {"type": "sources", "content": _format_sources(docs)}
