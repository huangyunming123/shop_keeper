from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

if __name__ == '__main__':

    file_title = "万用表使用说明"
    item_name = "RS-12数字万用表"

    bge_m3 = AIClients.get_bge_m3_client()
    result = generate_bge_m3_hybrid_vectors(bge_m3, [item_name])
    item_name_dense = result["dense"][0]
    item_name_sparse = result["sparse"][0]

    # 1.连接
    milvus_client = StorageClients.get_milvus_client()

    # 2.创建集合

    # 2.1 定义schema
    my_schema = milvus_client.create_schema()
    from pymilvus import DataType
    # 标量字段
    my_schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
    my_schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=256)
    my_schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=256)
    # 向量字段
    my_schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    my_schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 2.2 定义索引
    my_index_params = milvus_client.prepare_index_params()
    my_index_params.add_index(
        field_name="dense_vector",  # 建立索引的字段
        index_name="dense_vector_index",  # 索引名称
        index_type="AUTOINDEX",  # 索引类型  https://milvus.io/docs/zh/hnsw.md
        metric_type="COSINE",  # 向量度量类型(L2/IP/COSINE)
    )
    my_index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",  # IP 和 BM25
    )

    milvus_client.create_collection(
        collection_name="item_name_collection",
        schema=my_schema,
        index_params=my_index_params,
    )

    # 3.添加数据
    result = milvus_client.insert(
        collection_name="item_name_collection",
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": item_name_dense,
                "sparse_vector": item_name_sparse,
            }
        ]
    )
    print(result)
