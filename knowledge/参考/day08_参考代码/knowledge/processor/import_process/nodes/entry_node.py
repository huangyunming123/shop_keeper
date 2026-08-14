import json
import time
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState, get_default_state
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration


class EntryNode(BaseNode):

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.从state中获取参数: import_file_path, file_dir
        self.log_step("Step1","获取并校验State参数")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        # 2.校验参数
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="导入文件路径不能为空"
            )
        if not file_dir:
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="导入文件目录不能为空"
            )

        # 3.检查文件是否存在( from pathlib import Path )
        self.log_step("Step2","检查文件是否存在")
        path = Path(import_file_path)
        if not path.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="导入文件不存在"
            )

        # 4.判断文件类型
        self.log_step("Step3","判断文件类型")
        ext = path.suffix.lower()
        if ext == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif ext == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                message="不支持的文件类型"
            )

        # 5.获取文件标题
        state["file_title"] = path.stem   #  ..../RS-12万用表产品说明书.pdf

        return state



if __name__ == "__main__":
    state = get_default_state()
    state["import_file_path"] = r"E:\workspace\shopkeeper_brain\knowledge\processor\import_process\import_files\万用表RS-12的使用.pdf"
    state["file_dir"] = r"E:\workspace\shopkeeper_brain\knowledge\processor\import_process\import_files"

    node = EntryNode()
    state = node(state)

    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)