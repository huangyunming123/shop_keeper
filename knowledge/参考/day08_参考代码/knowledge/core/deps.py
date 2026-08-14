from functools import cache,lru_cache

from knowledge.services.upload_service import UploadService

# @cache   缓存注解（返回的实例会自动存入缓存，会出现OOM）
@lru_cache   # (淘汰策略：最近最少使用)
def get_upload_service():
    return UploadService()