import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Deque
from collections import deque
import time

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.import_prompt import IMG_SUMMARY_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class MdImgNode(BaseNode):

    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.校验state参数: 从state获取md文档内容、md文档路径对象、md文档图片目录
        md_content, md_path_obj, md_img_path_obj = self._get_md_content_and_path(state)
        # 2.扫描并过滤图片目录中的图片
        img_info_list = self._scan_and_filter_images(md_img_path_obj,md_content)
        # 3.调用VLM为每一个图片生成摘要（图片描述）
        image_summaries = self._generate_image_summaries(img_info_list)
        # 4.将图片保存到MinIO，获取在MinIO中的图片路径
        new_md_content = self._upload_img_and_update_md(md_path_obj.stem,md_content,img_info_list, image_summaries)
        # 5.将new_md_content保存到state中
        state["md_content"] = new_md_content
        return state

    def _get_md_content_and_path(self, state:ImportGraphState)-> Tuple[str,Path,Path]:
        """
        获取state中md文档的内容、md路径对象、md图片目录对象
        Args:
            state
        Returns:
            md文档的内容
            md路径对象
            md图片目录对象
        """
        # 1.检查md_path
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                node_name=self.name,
                field_name="md_path",
                message="md_path字段不存在"
            )
        # 2.读取md文档内容
        md_path_obj = Path(md_path)
        try:
            with open(md_path_obj,"r", encoding="utf-8") as f:
                md_content = f.read()
        except IOError as e:
            raise ImageProcessingError(
                node_name=self.name,
                message="读取md文档内容失败"
            )
        # 3.获取md文档的图片路径
        md_img_path_obj:Path = md_path_obj.parent / "images"
        return md_content, md_path_obj, md_img_path_obj

    def _scan_and_filter_images(self, md_img_path_obj:Path, md_content:str)->List[Tuple[str,str,Tuple[str,str,str]]]:
        img_info_list = []
        # 1.遍历图片目录中所有的图片
        for img_name in os.listdir(md_img_path_obj):
            # 2. 过滤图片后缀
            # img_name = "a.bmp"/"a.jpg"
            ext = os.path.splitext(img_name)[1]
            if ext not in self.config.image_extensions:
                continue
            # 3. 获取图片路径
            img_path = str(md_img_path_obj / img_name)
            # 4. 提取当前图片的上下文
            img_context = self._extract_img_context(img_name, md_content)
            # 5. 将（图片名称、图片路径、图片上下文）
            img_info_list.append((img_name, img_path, img_context))
        # 6. 返回图片信息列表
        return img_info_list

    def _extract_img_context(self, img_name:str, md_content:str)->Tuple[str,str,str]:
        """
        提取图片的上下文
        步骤：
            1.找到目标图片在md文档中的位置(行号)  47
               ![alt文本](imgs/347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg)
            2.从图片的位置开始向上找到最近的标题（标题行索引 37， 标题内容：安全标识 ）
            3.从图片的位置开始向下找最近的标题（标题行的索引 59）

        Args:
            img_name: 目标图片名称  "347706d8e5045d76f78334438c01c4b148a953dfe5c5f2f33b1fd269c1be2b1e.jpg"
            md_content: md文档内容

        Returns:
            元组(标题,上文,下文)
        """
        context_list = []
        # 1.找到目标图片在md文档中的位置(行号)
        md_lines = md_content.split("\n")
        # 定义图片正则表达式
        img_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        for index,line in enumerate(md_lines):
            if not img_pattern.search(line):
                continue
            img_index = index
            # 2.截取上文
            # 定义标题正则表达式
            title_pattern = re.compile(r"^#{1,6}\s+")
            # 从图片所在行的上一行开始，向上查找每一行，匹配到最近的标题行
            pre_title_index = -1
            pre_title_content = ""
            for i in range(img_index-1,-1,-1):
                if title_pattern.search(md_lines[i]):
                    pre_title_index = i
                    pre_title_content = md_lines[i]
                    break
            # 截取从pre_title_index的下一行到图片的上一行
            pre_context = "\n".join(md_lines[pre_title_index+1:img_index])
            final_pre_context = self._extract_context_with_limit(pre_context, 200, "up")

            # 3.截取下文
            post_title_index = -1
            for j in range(img_index+1,len(md_lines)):
                if title_pattern.search(md_lines[j]):
                    post_title_index = j
                    break
            post_context = "\n".join(md_lines[img_index+1:post_title_index])
            final_post_context =  self._extract_context_with_limit(post_context,200,"down")

            # 将标题、上文、下文组成一个元组，添加到列表中
            context_list.append( (pre_title_content,final_pre_context,final_post_context) )
        # 4.返回上下文列表
        if len(context_list) == 0:
            return ("","","")
        return context_list[0]

    def _extract_context_with_limit(self, context:str, max_chars:int,direction:str) ->str:
        """
        对上下文进行进一步的清洗和截断: 段落+字符限制
        Args:
            context:  上文/下文
            max_chars: 截取的最大字符数
            direction: 上下文标识（up标识上文，down标识下文）
        Returns:

        """
        # 1.将上文/下文转换为段落列表
        para_list = []
        current_para = []
        context_lines  = context.split("\n")
        for line in context_lines:
            striped_line = line.strip()
            if striped_line:
                # 如果当前行不为空，则通过正则判断是否为图片
                if re.match(r"^!\[.*?\]\(.*?\)$", striped_line):
                   if current_para:
                       p_str = "\n".join(current_para)
                       para_list.append(p_str)
                       current_para = []
                else:
                    current_para.append(striped_line)
            else:
                if current_para:
                    p_str = "\n".join(current_para)
                    para_list.append(p_str)
                    current_para = []
        # current_para中可能还有最后一个段落的内容
        if current_para:
            para_list.append("\n".join(current_para))

        # 2.截取段落
        selected_para_list = []

        if direction == "up":
            para_list.reverse()

        if len(para_list) > 0:
            selected_para_list.append(para_list[0])
            total_chars  = len(para_list[0])

            for p in para_list[1:]:
                len_p = len(p)
                if total_chars + len_p > max_chars:
                    break
                else:
                    selected_para_list.append(p)
                    total_chars += len_p

        if direction == "up":
            selected_para_list.reverse()

        return "\n\n".join(selected_para_list)

    def _generate_image_summaries(self, img_info_list)-> Dict[str,str]:
        """
        调用千问VLM生成图片的摘要
        Args:
            img_info_list:

        Returns:
            {
                img_name:summary,
                img_name:summary,
                img_name:summary,
                ...
            }
        """
        image_summaries = {}
        # 1.创建VLM客户端
        vlm_client = AIClients.get_vlm_client()
        # 2.遍历图片信息列表
        #request_timestamps: Deque[float] = deque()
        for img_name, img_path, img_context in img_info_list:
            # self.enforce_rate_limit(request_timestamps,8,60)
            summary = self._get_img_summary(vlm_client,img_path,img_context)
            image_summaries[img_name] = summary
        # 3.返回图片摘要
        return image_summaries

    def _get_img_summary(self, vlm_client, img_path:str, img_context:Tuple[str,str,str])->str:
        """
        调用VLM
        """

        # 1.构造提示词
        title, pre_context, post_context = img_context
        img_context = f"【图片上文】:｛pre_context｝\n【图片下文】:｛post_context｝"
        prompt_content = IMG_SUMMARY_PROMPT.format(
            title_content = title,
            img_context =  img_context,
        )

        # 2.读取图片的数据
        try:
            with open(img_path, "rb") as f:
                img_data = f.read()
                img_data_str = base64.b64encode(img_data).decode()
        except IOError as e:
            raise ImageProcessingError(f"图片文件读取失败: {e}")

        # 3.拼接完整提示词
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data_str}"
                        },
                    }
                ]
            }
        ]

        # 4.调用VLM
        completion = vlm_client.chat.completions.create(
            model=self.config.vl_model,
            messages=messages
        )
        # 5.返回生成的图片摘要
        return completion.choices[0].message.content.strip()

    def _upload_img_and_update_md(self, document_name, md_content, img_info_list, image_summaries):
        """
        Args:
            document_name:  文档名称
            md_content:  md文档内容
            img_info_list:  图片信息列表（img_name,img_path）
            image_summaries:  图片摘要
        Returns:
        """
        # 1.将图片上传至Minio
        minio_client = StorageClients.get_minio_client()
        # 2.遍历图片信息列表
        remote_urls = {}   # { a.jpg:http://..../a.jpg ,  b.jpg:http://..../b.jpg }
        for img_name, img_path, img_context in img_info_list:
            minio_client.fput_object(
                bucket_name=self.config.minio_bucket,
                object_name=f"{document_name}/{img_name}",
                file_path=img_path
            )
            remote_url = f"{self.config.get_minio_base_url()}/{self.config.minio_bucket}/{document_name}/{img_name}"
            remote_urls[img_name] = remote_url

        # image_summaries    { a.jpg:'a的摘要' ,  b.jpg:'b的摘要' }
        # md_content   ".........![](images/a.jpg)......."
        # new_md_content   ".........![a的摘要](http://..../a.jpg)......."
        new_md_content = md_content
        for img_name, summary in image_summaries.items():
            remote_url = remote_urls[img_name]
            # 定义当前图片的正则表达式
            replace_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
            new_md_content = replace_pattern.sub(f"![{summary}]({remote_url})", new_md_content)

        return new_md_content


    def enforce_rate_limit(self,
            request_timestamps: deque,  # 存储请求时间戳
            max_requests: int,  # 最大请求数
            window_seconds: int = 60  # 时间窗口（秒）
    ):
        current_time = time.time()

        # 移除窗口外的旧时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 如果达到上限，等待
        if len(request_timestamps) >= max_requests:
            sleep_time = window_seconds - (current_time - request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f"{self.name} is rate limited. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

        # 记录本次请求
        request_timestamps.append(time.time())

if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    state = {
        "task_id": "",
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "file_dir": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files",
        "import_file_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "pdf_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用.pdf",
        "md_path": "E:\\workspace\\shopkeeper_brain\\knowledge\\processor\\import_process\\import_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
        "file_title": "万用表RS-12的使用",
        "md_content": "",
        "chunks": [],
        "item_name": ""
    }

    node = MdImgNode()
    state = node(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)

