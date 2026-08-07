# -*- coding: utf-8 -*-
"""Streamlit 前端：文档管理 + 智能问答"""
import streamlit as st
from pathlib import Path

from app.services.document_service import ingest_document, list_documents, delete_document, ingest_directory
from app.services.rag_service import stream_search
from app.core.config import DOCS_DIR

st.set_page_config(page_title="企业知识库智能问答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def bootstrap_index():
    """启动时若知识库为空，自动重建 data/docs 中预置文档的索引（云端容器不持久）"""
    try:
        docs = list_documents("kb_default")
        if not docs:
            results = ingest_directory(collection_name="kb_default")
            return {"rebuilt": True, "files": len(results)}
        return {"rebuilt": False, "files": len(docs)}
    except Exception as e:
        return {"rebuilt": False, "error": str(e)}


_boot = bootstrap_index()

# ---------- 侧边栏：文档管理 ----------
with st.sidebar:
    st.header("📁 文档管理")
    collection = st.text_input("知识库名称", value="kb_default")

    uploaded = st.file_uploader(
        "上传文档（PDF/Word/Markdown/TXT）",
        type=["pdf", "md", "txt", "docx", "doc"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("📤 上传并入库", use_container_width=True):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            save_path = DOCS_DIR / f.name
            save_path.write_bytes(f.getvalue())
            try:
                result = ingest_document(save_path, collection)
                st.success(f"✅ {f.name}：{result['chunks']} 个分块已入库")
            except Exception as e:
                st.error(f"❌ {f.name}：{e}")

    st.divider()
    st.subheader("📄 知识库文档")
    docs = list_documents(collection)
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {doc}")
            if col2.button("删除", key=doc):
                count = delete_document(doc, collection)
                st.toast(f"已删除 {count} 个分块")
                st.rerun()
    else:
        st.info("知识库为空，请上传文档")

    st.divider()
    with st.expander("💡 使用说明"):
        st.markdown("""
1. 左侧上传文档（PDF/Word/Markdown）
2. 下方输入问题
3. 回答会附带来源引用
4. 无关问题会自动拒答
        """)

# ---------- 主区域：问答 ----------
st.title("📚 企业知识库智能问答")

if _boot.get("rebuilt"):
    st.info(f"知识库已自动初始化（{_boot.get('files', 0)} 个文档入库），可开始提问")
elif _boot.get("error"):
    st.warning(f"知识库初始化提示：{_boot['error']}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("请输入你的问题...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    sources = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for event in stream_search(prompt, collection):
            if event["type"] == "text":
                full += event["content"]
                placeholder.markdown(full + "▌")
            elif event["type"] == "sources":
                sources = event["content"]
        placeholder.markdown(full)

        if sources:
            st.markdown("---")
            st.caption("📎 引用来源：")
            for s in sources:
                page_info = f"（第 {s['page']} 页）" if s["page"] else ""
                st.caption(f"📄 {s['source']} {page_info}")

    st.session_state.messages.append({"role": "assistant", "content": full})
