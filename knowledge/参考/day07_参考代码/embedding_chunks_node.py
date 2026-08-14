import json
import logging
from typing import Any, Dict, List

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class EmbeddingChunksNode(BaseNode):

    name = "embedding_chunks_node"

    def process(self, state:ImportGraphState)->ImportGraphState:
        # 1. 校验参数
        chunks = self._validate_state(state)

        # 2. 获取BGE-M3文本嵌入模型的客户端
        bge_m3_client = AIClients.get_bge_m3_client()

        # 3. 调用BGE-M3模型进行文本嵌入
        dense_list = []
        sparse_list = []
        batch_count = self.config.embedding_batch_size
        for index in range(0,len(chunks),batch_count):
            batch_start_index = index   # 16
            batch_end_index = batch_start_index + batch_count
            if batch_end_index > len(chunks):
                batch_end_index = len(chunks)  # 20

            chunks_batch = chunks[batch_start_index:batch_end_index]

            # 对chunks_batch进行文本嵌入处理，生成稠密向量和稀疏向量
            chunks_batch_content = [c["content"] for c in chunks_batch]
            result = generate_bge_m3_hybrid_vectors(bge_m3_client,chunks_batch_content)
            #{
            #  dense: [[...],[...],[...],[...],[...],[...],[...],[..]],
            #  sparse: [{...},{...},{...},{...},{...},{...},{...},{...}]
            #}
            dense_list.extend(result["dense"])
            sparse_list.extend(result["sparse"])
            self.logger.info(f"处理完成：{batch_start_index+1} ~ {batch_end_index}")

        # 4. 将稠密向量和稀疏向量回填到chunks中
        for index,chunk in enumerate(chunks):
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]

        # 5. 将chunks回填到state中
        state["chunks"] = chunks
        return state

    def _validate_state(self, state:ImportGraphState)->List[Dict[str,Any]]:
        self.log_step("Step1","校验参数")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                message="chunks is required",
                expected_type= list
            )
        for index,chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunks",
                    message=f"[chunk_{index}] 不是期望的字典类型",
                    expected_type= dict
                )
        return chunks

if __name__ == '__main__':
    setup_logging(logging.DEBUG)
    # 生成一个带有20个chunks的state
    state = ImportGraphState()
    state["chunks"] = [{"content": f"chunk_{i}"} for i in range(20)]

    node = EmbeddingChunksNode()
    new_state = node.process(state)
    json_str = json.dumps(new_state, indent=4, ensure_ascii=False)
    print(json_str)