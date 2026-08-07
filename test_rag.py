# -*- coding: utf-8 -*-
"""命令行测试：python test_rag.py 验证 RAG 全链路"""
import sys
sys.path.insert(0, ".")

from app.core.config import DOCS_DIR
from app.services.document_service import ingest_directory, list_documents
from app.services.rag_service import search


def main():
    print("=" * 50)
    print("Step 1: 批量入库 data/docs 下的文档...")
    results = ingest_directory(DOCS_DIR)
    for r in results:
        print(f"  {r}")

    print("\nStep 2: 当前知识库文档:")
    for doc in list_documents():
        print(f"  - {doc}")

    print("\nStep 3: 测试问答:")
    questions = [
        "员工事假需要提前多久申请？",
        "年假怎么计算？",
        "旷工的处罚是什么？",
        "今天天气怎么样？",
    ]
    for q in questions:
        print(f"\n问: {q}")
        result = search(q)
        print(f"答: {result['answer']}")
        print(f"来源: {result['sources']}")


if __name__ == "__main__":
    main()
