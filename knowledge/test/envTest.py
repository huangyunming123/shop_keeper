"""
测试数据库连接和基础功能
包括：Milvus (向量数据库)、Neo4j (图数据库)、MongoDB (文档数据库)、MinIO (对象存储)
"""

import os
from dotenv import load_dotenv
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, MilvusException
from neo4j import GraphDatabase
from pymongo import MongoClient
from minio import Minio
from minio.error import S3Error


# 加载环境变量
load_dotenv()


def test_milvus():
    """
    测试 Milvus 向量数据库连接和基本操作
    """
    print("\n" + "="*60)
    print("测试 Milvus 向量数据库")
    print("="*60)
    
    try:
        # 获取配置
        milvus_url = os.getenv('MILVUS_URL')
        collection_name = os.getenv('CHUNKS_COLLECTION')
        
        print(f"✓ 连接地址：{milvus_url}")
        print(f"✓ 集合名：{collection_name}")
        
        # 创建连接
        from pymilvus import connections
        connections.connect('default', host='localhost', port=19530)
        print("✓ 成功连接到 Milvus")
        
        # 列出所有集合
        from pymilvus import utility
        all_collections = utility.list_collections()
        print(f"✓ 当前集合列表：{all_collections}")
        
        if collection_name in all_collections:
            # 如果集合已存在，进行测试
            collection = Collection(collection_name)
            print(f"✓ 成功加载集合：{collection_name}")
            print(f"✓ 数据量：{collection.num_entities}")
            
            # 获取集合信息
            schema = collection.schema
            print(f"\n📋 集合 schema:")
            for field in schema.fields:
                print(f"  - {field.name}: {field.dtype}")
        else:
            # 创建测试集合
            print(f"\n⚠️ 集合 {collection_name} 不存在，创建测试集合...")
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            
            schema_obj = CollectionSchema(fields, "Test Collection")
            collection = Collection("test_milvus_collection", schema_obj)
            print(f"✓ 创建测试集合成功：test_milvus_collection")
            
            # 插入测试数据
            import numpy as np
            test_data = [
                [np.random.rand(1536).tolist()],
                ["测试文本"],
                [{"source": "test"}]
            ]
            collection.insert(test_data)
            collection.flush()
            print(f"✓ 插入 1 条测试数据，当前数据量：{collection.num_entities}")
            
            # 查询测试
            # 创建索引并加载
            index_params = {
                "metric_type": "L2",
                "index_type": "FLAT",
                "params": {}
            }
            collection.create_index("embedding", index_params)
            collection.load()
            expr = "id > 0"
            result = collection.query(expr=expr, output_fields=["text"], limit=1)
            if result:
                print(f"✓ 查询结果：{result[0]['text']}")
            else:
                print(f"⚠️ 查询结果为空")
            
            # 清理测试集合
            utility.drop_collection("test_milvus_collection")
            print(f"✓ 清理测试集合完成")
        
        print("\n✅ Milvus 测试通过!")
        return True
        
    except MilvusException as e:
        print(f"\n❌ Milvus 连接失败：{e}")
        return False
    except Exception as e:
        print(f"\n❌ 发生未知错误：{e}")
        return False


def test_neo4j():
    """
    测试 Neo4j 图数据库连接和基本操作
    """
    print("\n" + "="*60)
    print("测试 Neo4j 图数据库")
    print("="*60)
    
    try:
        # 获取配置
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD')
        database = os.getenv('NEO4J_DATABASE')
        
        print(f"✓ URI: {uri}")
        print(f"✓ 用户名：{username}")
        print(f"✓ 数据库：{database}")
        
        # 创建连接
        driver = GraphDatabase.driver(uri, auth=(username, password))
        print("✓ 成功连接到 Neo4j")
        
        # 测试连接
        with driver.session(database=database) as session:
            # 运行 Cypher 查询
            result = session.run("RETURN 1 AS test")
            record = result.single()
            print(f"✓ 查询测试：{record['test']}")
            
            # 获取数据库信息
            version_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
            version = version_result.single()['version']
            print(f"✓ Neo4j 版本：{version}")
            
            # 创建测试节点
            session.run("""
                CREATE (t:TestNode {id: $id, createTime: timestamp()})
                RETURN t
            """, id="neo4j_test_001")
            print("✓ 成功创建测试节点")
            
            # 查询测试节点
            result = session.run(
                "MATCH (t:TestNode {id: $id}) RETURN t",
                id="neo4j_test_001"
            )
            node = result.single()
            if node:
                print(f"✓ 查询到测试节点：{node['t'].get('id')}")
                
                # 删除测试节点
                session.run(
                    "MATCH (t:TestNode {id: $id}) DELETE t",
                    id="neo4j_test_001"
                )
                print("✓ 删除测试节点完成")
        
        driver.close()
        print("\n✅ Neo4j 测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ Neo4j 连接失败：{e}")
        return False


def test_mongodb():
    """
    测试 MongoDB 文档数据库连接和基本操作
    """
    print("\n" + "="*60)
    print("测试 MongoDB 文档数据库")
    print("="*60)
    
    try:
        # 获取配置
        mongo_url = os.getenv('MONGO_URL')
        db_name = os.getenv('MONGO_DB_NAME')
        
        print(f"✓ 连接 URL: {mongo_url}")
        print(f"✓ 数据库名：{db_name}")
        
        # 创建连接
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # 测试连接
        client.server_info()
        print("✓ 成功连接到 MongoDB")
        
        # 获取 MongoDB 版本
        version = client.server_info()['version']
        print(f"✓ MongoDB 版本：{version}")
        
        # 获取或创建数据库
        db = client[db_name]
        print(f"✓ 连接到数据库：{db_name}")
        
        # 获取或创建集合
        collection_name = f"{db_name}_test"
        collection = db[collection_name]
        
        # 插入测试数据
        test_doc = {
            "name": "MongoDB Test",
            "data": {
                "timestamp": "2024-08-14T12:00:00Z",
                "status": "active"
            },
            "tags": ["test", "mongodb", "database"]
        }
        result = collection.insert_one(test_doc)
        print(f"✓ 插入测试文档：{result.inserted_id}")
        
        # 查询测试
        doc = collection.find_one({"name": "MongoDB Test"})
        if doc:
            print(f"✓ 查询到文档：{doc['name']}")
            
            # 更新测试
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"status": "updated"}}
            )
            print("✓ 更新文档成功")
        
        # 删除测试文档
        collection.delete_one({"name": "MongoDB Test"})
        print("✓ 删除测试文档完成")
        
        # 关闭连接
        client.close()
        print("\n✅ MongoDB 测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ MongoDB 连接失败：{e}")
        return False


def test_minio():
    """
    测试 MinIO 对象存储连接和基本操作
    """
    print("\n" + "="*60)
    print("测试 MinIO 对象存储")
    print("="*60)
    
    try:
        # 获取配置
        endpoint = os.getenv('MINIO_ENDPOINT')
        access_key = os.getenv('MINIO_ACCESS_KEY')
        secret_key = os.getenv('MINIO_SECRET_KEY')
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        
        print(f"✓ Endpoint: {endpoint}")
        print(f"✓ 访问密钥：{access_key}")
        print(f"✓ 存储桶：{bucket_name}")
        
        # 创建客户端
        minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        print("✓ 成功创建 MinIO 客户端")
        
        # 检查存储桶是否存在
        if not minio_client.bucket_exists(bucket_name):
            print(f"⚠️ 存储桶 {bucket_name} 不存在，创建中...")
            minio_client.make_bucket(bucket_name)
            print(f"✓ 创建存储桶成功：{bucket_name}")
        else:
            print(f"✓ 存储桶存在：{bucket_name}")
        
        # 测试上传对象
        import io
        test_data = b"Hello, MinIO! This is a test file."
        object_name = "test/test-file.txt"
        test_data_stream = io.BytesIO(test_data)
        
        minio_client.put_object(
            bucket_name,
            object_name,
            test_data_stream,
            len(test_data),
            content_type="text/plain"
        )
        print(f"✓ 上传测试文件成功：{object_name}")
        
        # 测试下载对象
        response = minio_client.get_object(bucket_name, object_name)
        downloaded_data = response.read()
        print(f"✓ 下载测试文件成功，大小：{len(downloaded_data)} bytes")
        
        # 验证内容
        if downloaded_data == test_data:
            print("✓ 文件内容验证通过")
        else:
            print("⚠️ 文件内容不匹配")
        
        # 列出文件
        objects = list(minio_client.list_objects(bucket_name, prefix="test/", recursive=True))
        print(f"✓ 测试目录下共有 {len(objects)} 个文件")
        
        # 测试删除
        minio_client.remove_object(bucket_name, object_name)
        print(f"✓ 删除测试文件成功：{object_name}")
        
        print("\n✅ MinIO 测试通过!")
        return True
        
    except S3Error as e:
        print(f"\n❌ MinIO S3 错误：{e}")
        return False
    except Exception as e:
        print(f"\n❌ 发生未知错误：{e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("开始数据库连接测试")
    print("="*60)
    
    results = {}
    
    # 测试各个组件
    results['Milvus'] = test_milvus()
    results['Neo4j'] = test_neo4j()
    results['MongoDB'] = test_mongodb()
    results['MinIO'] = test_minio()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for component, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {component}: {status}")
    
    print(f"\n总计：{passed}/{total} 个组件测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确！")
    else:
        print(f"\n⚠️ 还有 {total - passed} 个组件需要检查")
