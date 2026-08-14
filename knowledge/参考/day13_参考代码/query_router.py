import asyncio
from functools import partial
from typing import Union

import uvicorn
from fastapi import FastAPI,Depends,BackgroundTasks,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryResponse, StreamSubmitResponse, QueryRequest, HistoryResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_result

# 1. 创建FastAPI
app = FastAPI(description="", version="v1.0")

# 2. 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 挂载静态文件
static_resource_page_dir = get_front_page_dir()
if static_resource_page_dir:
    app.mount("/chat", StaticFiles(directory=static_resource_page_dir))

@app.post("/query",response_model=Union[StreamSubmitResponse, QueryResponse])
async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        query_service:QueryService = Depends(get_query_service)
):
    session_id = request.session_id
    is_stream = request.is_stream
    query = request.query
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(session_id,query,is_stream)
    # 1.如果session_id不存在，则创建一个session_id
    if not session_id:
        session_id = query_service.generate_session_id()
    # 2.创建一个任务ID
    task_id = query_service.generate_task_id()
    # 3.启动查询流程
    if is_stream:
        # 如果是流式调用，则先创建sse队列
        create_sse_queue(task_id)
        # 后台任务启动查询流程
        background_tasks.add_task(query_service.run_query_graph, task_id, query, session_id,is_stream)
        return StreamSubmitResponse(
            message="正在执行查询",
            session_id=session_id,
            task_id=task_id
        )
    else:
        # 获取到调用当前async的事件循环对象
        event_loop = asyncio.get_event_loop()
        # 将函数放到事件循环中执行
        func_with_args = partial(
            query_service.run_query_graph,
            task_id,
            query,
            session_id,
            is_stream
        )
        # 在循环事件中执行函数
        await event_loop.run_in_executor(None, func_with_args)
        # 获取结果
        answer = query_service.get_task_result(task_id)
        # answer = query_service.run_query_graph(task_id, query, session_id,is_stream)
        return QueryResponse(
            message="查询成功",
            session_id=session_id,
            answer=answer
        )

@app.get("/stream/{task_id}")
async def stream(task_id:str,request:Request):
    return StreamingResponse(
        content=sse_generator(task_id,request),
        media_type="text/event-stream"
    )

@app.get("/status/{task_id}")
async def status(task_id:str):
    # 查询任务状态
    return {
      "status": "processing",
      "done_list": [],
      "running_list": [],
      "durations": {},
      "result":"answer"
    }

@app.get("/history/{session_id}",response_model=HistoryResponse)
async def get_history(
        session_id:str,
        query_service:QueryService = Depends(get_query_service)
):
    # 获取历史记录
    history_list = query_service.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        items=history_list
    )

@app.delete("/history/{session_id}")
async def clear_history(
        session_id:str,
        query_service:QueryService = Depends(get_query_service)
):
    # 清空当前会话的历史记录
    deleted_count = query_service.delete_history(session_id)
    return {
      "message": "历史记录清理完成",
      "deleted_count": deleted_count
    }


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8001, log_level="info")