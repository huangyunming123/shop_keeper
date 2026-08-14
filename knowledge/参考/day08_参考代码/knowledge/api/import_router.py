import json

import uvicorn
from fastapi import FastAPI,UploadFile,Depends,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_upload_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.upload_service import UploadService

# 1.创建FastAPI实例
app = FastAPI(description="掌柜智库文档导入服务",version="2.1.23")

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,      # ← 默认值，可以省略
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态资源
page_path = get_front_page_dir()
if page_path:
    app.mount("/front", StaticFiles(directory=page_path))


# 2.定义路由: 定义文件上传接口
@app.post("/upload",response_model=UploadResponse)  # 注解
def upload_file(
        file:UploadFile,
        background_tasks:BackgroundTasks,  # background_tasks就是一个后台任务管理器
        upload_service:UploadService = Depends(get_upload_service)
):

    # 调用UploadService类中定义process_upload_file方法处理上传文件
    message, task_id,import_file_path,file_dir = upload_service.process_upload_file(file)

    # 文档处理流程需要花费较长的时间
    # 我们将耗时较长的业务放在后任务中执行，返回给用户一个结果
    background_tasks.add_task(
        upload_service.run_import_graph,
        import_file_path,file_dir,task_id
    )

    # 返回结果
    return UploadResponse(
        message=message,
        task_id=task_id
    )


# 3.定义路由: 定义任务状态查询接口
@app.get("/status/{task_id}",response_model=TaskStatusResponse)
def status(
        task_id:str,
        upload_service:UploadService = Depends(get_upload_service)
):
    result = upload_service.get_status(task_id)
    json_str = json.dumps(result, ensure_ascii=False,indent=4)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(json_str)
    return result



if __name__ == "__main__":
    # 4.启动服务
    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")