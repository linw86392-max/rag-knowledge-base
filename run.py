# -*- coding: utf-8 -*-
"""项目入口：python run.py 启动 FastAPI 服务"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
