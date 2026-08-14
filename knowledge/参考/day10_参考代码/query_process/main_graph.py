import json
import logging

from langgraph.graph import StateGraph

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.nodes.hybrid_vector_search_node import HybridVectorSearchNode
from knowledge.processor.query_process.nodes.hyde_vector_search_node import HyDEVectorSearchNode
from knowledge.processor.query_process.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_process.nodes.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_process.state import QueryGraphState, get_default_state


def my_router(state:QueryGraphState):
    if state.get("answer"):
        return True
    else:
        return False


def create_query_graph()-> StateGraph:
    graph = StateGraph(QueryGraphState)
    # 添加节点
    graph.add_node("item_name_confirmed_node", ItemNameConfirmedNode())
    # 添加多路召回的虚拟节点
    graph.add_node("multi_search", lambda x: x)
    graph.add_node("hybrid_vector_search_node", HybridVectorSearchNode())
    graph.add_node("hyde_vector_search_node",HyDEVectorSearchNode())
    graph.add_node("web_mcp_search_node",WebMcpSearchNode())
    # 添加一个汇聚的虚拟节点
    graph.add_node("join_node", lambda x: {})
    # ....

    # 添加边
    graph.add_edge("__start__", "item_name_confirmed_node")
    graph.add_conditional_edges("item_name_confirmed_node", my_router,{
        True:"__end__",
        False:"multi_search"
    } )
    # 从multi_search分发到三路召回节点
    graph.add_edge("multi_search","hybrid_vector_search_node")
    graph.add_edge("multi_search","hyde_vector_search_node")
    graph.add_edge("multi_search","web_mcp_search_node")
    # 三路召回节点汇聚到join_node
    graph.add_edge("hybrid_vector_search_node","join_node")
    graph.add_edge("hyde_vector_search_node","join_node")
    graph.add_edge("web_mcp_search_node","join_node")
    # ...
    graph.add_edge("join_node","__end__")

    return graph.compile()

if __name__ == '__main__':
    setup_logging(logging.INFO)
    graph:StateGraph = create_query_graph()
    state = get_default_state()
    state["original_query"] = "RS-12数字万用表如何测量电阻？"
    state = graph.invoke(state)
    json_str = json.dumps(state,indent=4,ensure_ascii=False)
    print(json_str)

