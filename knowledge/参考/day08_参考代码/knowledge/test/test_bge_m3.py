from modelscope import snapshot_download
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.ai_clients import AIClients

local_dir = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir=r"E:\ai_models\modelscope_cache\models\BAAI\bge-m3"
)

if __name__ == '__main__':

    # 1. 加载模型
    bge_m3 = AIClients.get_bge_m3_client()

    # 2. 生成嵌入
    response = bge_m3.encode_documents(["RS-12 数字万用表"])

    # 3. 提取向量
    dense_vector = response["dense"][0].tolist()  # List[float], 长度 1024
    sparse_matrix = response["sparse"]  # CSR 稀疏矩阵

    # 4. 从 CSR 矩阵提取稀疏向量
    start_idx = sparse_matrix.indptr[0]   #   start_idx = 0
    end_idx = sparse_matrix.indptr[1]     #   end_idx = 7
    token_ids = sparse_matrix.indices[start_idx:end_idx].tolist()  # [6,1773,....]
    weights = sparse_matrix.data[start_idx:end_idx].tolist()       # [0.00083,0.13325...]
    sparse_vector = dict(zip(token_ids, weights))  # Dict[int, float]