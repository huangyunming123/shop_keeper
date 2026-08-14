from typing import List, Dict

from pydantic import BaseModel,Field

class UploadResponse(BaseModel):
    message:str = Field(..., description="提示信息")
    task_id:str = Field(..., description="任务 ID，用于任务追踪(web交互的时候用到，实时看到节点的处理日志)")


class TaskStatusResponse(BaseModel):
    status:str = Field(..., description="任务状态")
    done_list:List[str] = Field(..., description="已完成的节点列表")
    running_list:List[str] = Field(..., description="正在运行的节点列表")
    durations:Dict[str,float] = Field(..., description="节点处理时间")