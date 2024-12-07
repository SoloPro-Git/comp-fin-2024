from pydantic import BaseModel
from typing import Optional, List

class TableSearchRequest(BaseModel):
    keyword: str
    fields: List[str]         # 要搜索的字段列表
    index_names: List[str]    # 支持多个索引名称

class TableSearchResponse(BaseModel):
    keyword: str
    score: float
    index_name: str          # 添加索引名称，用于区分数据来源
    source_data: dict