from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence
from pymilvus import CollectionSchema, FieldSchema, DataType,MilvusClient

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


class MilvusImportNode(BaseNode):

    name = "milvus_import_node"

    def process(self, state:ImportGraphState)->ImportGraphState:
        # 1.校验参数
        chunks = self._validate_state(state)
        
        # 2. 获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        
        # 3. 创建一个存储chunks的集合
        collection_name = self.config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self._create_chunks_collection(milvus_client, collection_name)

        # 4. 将chunks存储到Milvus中
        chunks = _MilvusInserter(milvus_client,collection_name).insert_rows(chunks)

        state["chunks"] = chunks
        return state

    def _validate_state(self, state:ImportGraphState)->List[Dict[str,Any]]:
        chunks = state.get("chunks")
        # 1.非空和类型验证
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                message="chunks is required",
                expected_type= list
            )
        # 2.校验每个chunk,必须是字典类型，必须包含稠密向量和稀疏向量
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunks",
                    message="chunks must be a list of dicts",
                    expected_type= dict
                )
            if "dense_vector" not in chunk or "sparse_vector" not in chunk:
                raise StateFieldError(
                    node_name=self.name,
                    field_name="chunks",
                    message="chunks must contain dense_vector and sparse_vector",
                    expected_type= dict
                )
        # 3.返回
        return chunks

    def _create_chunks_collection(self, milvus_client, collection_name):
        schema = _MilvusSchemaBuilder.build_schema(milvus_client)
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        # 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"Milvus集合{collection_name}创建成功")

@dataclass
class _SCALAR_FIELD_SPC:
    field_name:str
    datatype:DataType
    max_length:Optional[int] = None

# 标量字段列表
_SCALAR_FIELDS : Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name",datatype=DataType.VARCHAR,max_length=256)
)

class _MilvusSchemaBuilder:
    @staticmethod
    def build_schema(milvus_client: MilvusClient) -> CollectionSchema:
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        # 添加主键字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 添加标量字段
        for scalar_field in _SCALAR_FIELDS:
            args: Dict[str, Any] = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length:
                args["max_length"] = scalar_field.max_length
            schema.add_field(**args)
        # 添加向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        return schema

class _MilvusIndexBuilder:
    @staticmethod
    def build_index_params(milvus_client: MilvusClient) -> Dict[str, Any]:
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        return index_params

class _MilvusInserter:

    def __init__(self,milvus_client:MilvusClient,collection_name):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def insert_rows(self,chunks:List[Dict[str,Any]]):
        # 1.执行插入操作
        insert_results = self.milvus_client.insert( self.collection_name,data=chunks )
        # 2.将自动生成id回填到chunks中
        chunk_ids = insert_results.get('ids')  # [1231,123123,12312..]
        for id,chunk in zip(chunk_ids,chunks):
            chunk["chunk_id"] = id
        # 3.返回
        return chunks