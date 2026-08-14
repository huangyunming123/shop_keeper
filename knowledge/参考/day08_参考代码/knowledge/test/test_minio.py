import os

from minio import Minio
from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "192.168.44.101:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False
    )

    if not client.bucket_exists("my-bucket"):
        client.make_bucket("my-bucket")

    buckets = client.list_buckets()
    bucket_names = [b.name for b in buckets]

    # 上传文件
    client.fput_object(
        "my-bucket",  # Bucket 名称
        "images/a.jpg",  # Object 名称（含路径）
        r"E:\workspace\shopkeeper_brain\knowledge\processor\import_process\import_files\万用表RS-12的使用\auto\images\01ff135dc95789f7cb428c34df92a77869db4f4e70b83d663d1c485a17e416c1.jpg",  # 本地文件路径
        content_type="image/jpeg"  # MIME 类型
    )


    print(f"  ✓ MinIO 连接成功，存储桶: {bucket_names}")