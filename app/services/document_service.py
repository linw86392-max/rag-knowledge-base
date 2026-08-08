# -*- coding: utf-8 -*-
"""文档处理：加载 -> 分块 -> 向量化入库"""
import uuid
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.core.config import DOCS_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.core.models import get_embeddings


def load_document(file_path: Path):
    """按扩展名选择加载器，返回 Documents（含来源元数据）"""
    suffix = file_path.suffix.lower()
    source = str(file_path)

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(source)
    elif suffix in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(source, encoding="utf-8")
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(source)
    elif suffix == ".doc":
        raise ValueError(".doc 老格式请先在 Word 中另存为 .docx 再上传")
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = file_path.name  # 只存文件名，便于溯源显示
    return docs


def split_documents(docs):
    """语义分块：500字 + 100字重叠，保留段落边界"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_vectorstore(collection_name: str = "kb_default"):
    """获取/创建向量库实例（持久化到本地 Chroma）"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def ingest_document(file_path: Path, collection_name: str = "kb_default") -> dict:
    """单文档入库全流程，返回统计"""
    docs = load_document(file_path)
    chunks = split_documents(docs)
    for c in chunks:
        c.metadata["chunk_id"] = str(uuid.uuid4())

    vectorstore = get_vectorstore(collection_name)
    ids = vectorstore.add_documents(chunks)

    return {
        "file": file_path.name,
        "total_pages": len(docs),
        "chunks": len(ids),
    }


def ingest_directory(directory: Path = DOCS_DIR, collection_name: str = "kb_default") -> List[dict]:
    """批量入库目录下所有支持的文件"""
    results = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in (".pdf", ".md", ".txt", ".docx", ".doc"):
            try:
                results.append(ingest_document(f, collection_name))
            except Exception as e:
                results.append({"file": f.name, "error": str(e)})
    return results


def delete_document(filename: str, collection_name: str = "kb_default") -> int:
    """按文件名删除文档向量"""
    vectorstore = get_vectorstore(collection_name)
    chunks = vectorstore.get(where={"source": filename})
    if not chunks["ids"]:
        return 0
    vectorstore.delete(ids=chunks["ids"])
    return len(chunks["ids"])


def list_documents(collection_name: str = "kb_default") -> List[str]:
    """列出知识库中所有文档"""
    vectorstore = get_vectorstore(collection_name)
    chunks = vectorstore.get(include=["metadatas"])
    sources = set()
    for meta in chunks.get("metadatas", []):
        if meta and meta.get("source"):
            sources.add(meta["source"])
    return sorted(sources)
