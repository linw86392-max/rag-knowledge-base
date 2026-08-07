# -*- coding: utf-8 -*-
"""FastAPI 入口：文档管理 + RAG 问答（SSE 流式）"""
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core.config import DOCS_DIR
from app.services import document_service, rag_service

app = FastAPI(title="企业知识库智能问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/kb/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Query("kb_default"),
):
    """上传文档并建立索引"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DOCS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        result = document_service.ingest_document(save_path, collection)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/kb/ingest")
def ingest_folder(collection: str = Query("kb_default")):
    """批量入库 data/docs 目录下所有文档"""
    results = document_service.ingest_directory(collection_name=collection)
    return {"success": True, "results": results}


@app.get("/api/kb/list")
def list_docs(collection: str = Query("kb_default")):
    """知识库文档列表"""
    return {"documents": document_service.list_documents(collection)}


@app.delete("/api/kb/document/{filename}")
def delete_doc(filename: str, collection: str = Query("kb_default")):
    """删除文档及向量"""
    count = document_service.delete_document(filename, collection)
    return {"deleted": count}


@app.post("/api/qa/ask")
async def ask(question: str = Query(...), collection: str = Query("kb_default")):
    """非流式问答（测试用）"""
    result = rag_service.search(question, collection)
    return result


@app.post("/api/qa/ask/stream")
async def ask_stream(question: str = Query(...), collection: str = Query("kb_default")):
    """SSE 流式问答"""
    def gen():
        yield _sse({"type": "start"})
        for event in rag_service.stream_search(question, collection):
            yield _sse(event)
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
