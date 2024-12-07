from elasticsearch import Elasticsearch
from typing import List
from app.models import TableSearchResponse, TableSearchRequest

class ESService:
    def __init__(self, es_host: str = "localhost", es_port: int = 9200):
        self.es_client = Elasticsearch(
            [f'http://{es_host}:{es_port}'],
            basic_auth=('elastic', 'elastic')
        )

    def search_tables(self, search_request: TableSearchRequest) -> List[TableSearchResponse]:
        query = {
            "query": {
                "multi_match": {
                    "query": search_request.keyword,
                    "fields": search_request.fields,
                    "type": "best_fields",
                    "tie_breaker": 0.3
                }
            },
            "explain": True
        }

        response = self.es_client.search(
            index=",".join(search_request.index_names),
            body=query
        )

        # 收集所有结果
        results = []
        seen_data = set()  # 用于存储已经见过的数据的指纹
        
        for hit in response['hits']['hits']:
            # 将source_data转换为元组，用于创建指纹
            source_items = tuple(sorted(hit['_source'].items()))
            
            # 如果这个数据还没见过，就添加到结果中
            if source_items not in seen_data:
                seen_data.add(source_items)
                results.append(
                    TableSearchResponse(
                        keyword=search_request.keyword,
                        score=hit['_score'],
                        index_name=hit['_index'],
                        source_data=hit['_source']
                    )
                )
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results 