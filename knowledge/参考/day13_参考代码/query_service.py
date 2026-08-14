import logging
import uuid
from typing import Any, Dict, List

from langgraph.graph import StateGraph

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.main_graph import create_query_graph
from knowledge.processor.query_process.state import get_default_state
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history
from knowledge.utils.task_util import get_task_result


class QueryService:

    @staticmethod
    def generate_session_id()->str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id()->str:
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self,task_id:str, query: str, session_id: str,is_stream:bool):
        setup_logging(logging.INFO)
        # 创建查询图
        graph: StateGraph = create_query_graph()
        # 初始化状态
        state = get_default_state()
        state["session_id"] = session_id
        state["task_id"] = task_id
        state["original_query"] = query
        state["is_stream"] = is_stream
        # 调用查询图
        state = graph.invoke(state)
        return state["answer"]

    def get_task_result(self, task_id: str) -> str:
        return get_task_result(task_id=task_id, key="answer")

    def get_history(self,session_id:str)->List[Dict[str,Any]]:
        history_list =  get_recent_messages(session_id,20)
        return history_list

    def delete_history(self,session_id:str)->int:
        deleted_count = clear_history(session_id)
        return deleted_count