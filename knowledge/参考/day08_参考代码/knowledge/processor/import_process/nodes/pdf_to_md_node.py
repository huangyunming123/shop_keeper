import subprocess
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.校验state参数
        pdf_path_obj, file_dir_obj = self._validate_state(state)
        # 2.调用MinerU工具解析PDF：创建一个进程执行mineru命令
        process_code = self._execute_mineru(pdf_path_obj,file_dir_obj)
        # 3.获取转换生成的md文档路径
        md_path = self._get_md_path(pdf_path_obj,file_dir_obj)
        # 4.更新state
        state["md_path"] = md_path
        # 5.返回state
        return state


    def _validate_state(self, state: ImportGraphState) -> Tuple[Path,Path]:
        self.log_step("Step1","校验参数")
        pdf_path = state.get("pdf_path")
        file_dir = state.get("file_dir")
        # pdf_path_obj 是PDF文件的Path对象
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="pdf_path",
                message="PDF文件不存在"
            )
        # file_dir_obj 是文件输出目录的Path对象
        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="文件输出目录不存在"
            )
        # 返回
        return pdf_path_obj,file_dir_obj

    def _execute_mineru(self, pdf_path_obj:Path, file_dir_obj:Path)-> int:
        self.log_step("Step2","执行MinerU")
        # 1.构建cmd指令： mineru -p <pdf文档> -o <输出目录> -b pipeline --source local
        cmd = [
            "mineru",
            "-p", str(pdf_path_obj),
            "-o", str(file_dir_obj),
            "-b",  "pipeline",
            "--source", "local"
        ]
        # 2.执行指令 (import subprocess)
        start_time = time.time()
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        # 3.获取命令执行的日志
        for line in process.stdout:
            self.logger.info(f'MinerU: {line.strip()}')

        process_code = process.wait()
        end_time = time.time()
        if process_code != 0:
            raise PdfConversionError(
                node_name=self.name,
                message="MinerU转换失败"
            )
        else:
            self.logger.info(f"MinerU转换完成，耗时：{end_time - start_time:.2f}秒")
        return process_code

    def _get_md_path(self, pdf_path_obj:Path, file_dir_obj:Path)->str:
        self.log_step("Step3","获取md文件路径")
        # 1.提取pdf文件名
        pdf_stem = pdf_path_obj.stem
        # 2.拼接md文件路径
        md_path = file_dir_obj / pdf_stem/"auto"/f"{pdf_stem}.md"
        if not md_path.exists():
            raise PdfConversionError(
                node_name=self.name,
                message="MinerU转换失败"
            )
        #  3.返回
        return str(md_path)


if __name__ == "__main__":
    import json
    state = {
        "task_id": "",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "file_dir": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files",
        "import_file_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "pdf_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "md_path": "",
        "file_title": "万用表RS-12的使用",
        "md_content": "",
        "chunks": [],
        "item_name": ""
    }

    node = PdfToMdNode()
    state = node(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)

    # {
    #     "task_id": "",
    #     "is_pdf_read_enabled": true,
    #     "is_md_read_enabled": false,
    #     "file_dir": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files",
    #     "import_file_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
    #     "pdf_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
    #     "md_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
    #     "file_title": "万用表RS-12的使用",
    #     "md_content": "",
    #     "chunks": [],
    #     "item_name": ""
    # }