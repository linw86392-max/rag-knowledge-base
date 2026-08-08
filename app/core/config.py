# -*- coding: utf-8 -*-
"""全局配置：模型、向量库、路径"""
import os
from pathlib import Path


def _get_env(key: str, default: str = ""):
    """先读环境变量，读不到再读 Streamlit Secrets（云端 Settings -> Secrets）。
    本地脚本（非 Streamlit 运行时）也能用，Streamlit 不可用时回退环境变量。"""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


# 路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # app/core/config.py -> 项目根目录
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# LLM 配置：ollama（本地）/ dashscope（通义）/ deepseek / siliconflow
LLM_PROVIDER = _get_env("LLM_PROVIDER", "deepseek")  # ollama / dashscope / deepseek / siliconflow

# Embedding 配置：ollama（本地免费）/ dashscope（通义）/ siliconflow（硅基流动）
# 注意：DeepSeek 不提供 embedding，用 deepseek 时默认 siliconflow 向量化
EMBEDDING_PROVIDER = _get_env("EMBEDDING_PROVIDER", "siliconflow")

# Ollama 本地模型
OLLAMA_BASE_URL = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _get_env("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBEDDING_MODEL = _get_env("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")

# 通义千问（阿里云百炼）
DASHSCOPE_API_KEY = _get_env("DASHSCOPE_API_KEY", "sk-你的key")
DASHSCOPE_MODEL = _get_env("DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_EMBEDDING_MODEL = _get_env("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v2")

# DeepSeek
DEEPSEEK_API_KEY = _get_env("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = _get_env("DEEPSEEK_MODEL", "deepseek-chat")

# 硅基流动 SiliconFlow（OpenAI 兼容，LLM + Embedding）
SILICONFLOW_API_KEY = _get_env("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = _get_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = _get_env("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SILICONFLOW_EMBEDDING_MODEL = _get_env("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 检索参数
TOP_K = 5
RERANK_THRESHOLD = 0.5  # 相关性阈值，低于则拒答
