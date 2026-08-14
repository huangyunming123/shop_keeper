import logging

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState, get_default_state


def create_import_graph()-> StateGraph:
    # 1.创建状态图
    graph = StateGraph(ImportGraphState)

    # 2.添加节点
    graph.add_node("entry_node",EntryNode())
    graph.add_node("pdf_to_md_node",PdfToMdNode())

    # 3.定义边
    graph.add_edge("__start__","entry_node")
    graph.add_edge("entry_node","pdf_to_md_node")
    graph.add_edge("pdf_to_md_node","__end__")

    # 4.编译状态图
    return graph.compile()

if __name__ == "__main__":
    # 开启日志
    setup_logging(logging.DEBUG)

    graph = create_import_graph()

    state = get_default_state()
    state["import_file_path"] = r"E:\workspace\shopkeeper_brain\knowledge\processor\import_process\import_files\hak180产品安全手册.pdf"
    state["file_dir"] = r"E:\workspace\shopkeeper_brain\knowledge\processor\import_process\import_files"

    state = graph.invoke(state)

