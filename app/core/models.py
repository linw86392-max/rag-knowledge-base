# -*- coding: utf-8 -*-
"""模型工厂：统一返回 ChatModel 与 Embedding"""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek

from app.core.config import (
    LLM_PROVIDER,
    EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    DASHSCOPE_API_KEY,
    DASHSCOPE_MODEL,
    DASHSCOPE_EMBEDDING_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    SILICONFLOW_EMBEDDING_MODEL,
)


def get_llm():
    """返回大模型聊天实例"""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.3,
            num_ctx=8192,
        )
    if LLM_PROVIDER == "deepseek":
        return ChatDeepSeek(
            api_key=DEEPSEEK_API_KEY,
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            timeout=60,
        )
    if LLM_PROVIDER == "siliconflow":
        return ChatOpenAI(
            api_key=SILICONFLOW_API_KEY,
            model=SILICONFLOW_MODEL,
            base_url=SILICONFLOW_BASE_URL,
            temperature=0.3,
            timeout=60,
        )
    # 默认通义千问（阿里云百炼，OpenAI 兼容端点）
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        model=DASHSCOPE_MODEL,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
        timeout=60,
    )


def get_embeddings():
    """返回 Embedding 模型实例（与 LLM 独立配置）"""
    if EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL,
        )
    if EMBEDDING_PROVIDER == "siliconflow":
        return OpenAIEmbeddings(
            api_key=SILICONFLOW_API_KEY,
            model=SILICONFLOW_EMBEDDING_MODEL,
            base_url=SILICONFLOW_BASE_URL,
        )
    return OpenAIEmbeddings(
        api_key=DASHSCOPE_API_KEY,
        model=DASHSCOPE_EMBEDDING_MODEL,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
