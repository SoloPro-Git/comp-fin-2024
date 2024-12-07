from elasticsearch import Elasticsearch

# 创建ES客户端连接
es = Elasticsearch(
    ['http://localhost:9200'],
    basic_auth=('elastic', 'elastic')
)

def search_tables(keyword=None):
    """查询库表关系"""
    index = '库表关系'
    
    if keyword:
        query = {
            "query": {
                "multi_match": {
                    "query": keyword,
                    "fields": ["*"]  # 搜索所有字段
                }
            }
        }
    else:
        query = {"query": {"match_all": {}}}
    
    result = es.search(index=index, body=query, size=100)
    return result['hits']['hits']

def search_fields(table_name=None):
    """查询表字段信息"""
    index = '表字段信息'
    
    if table_name:
        query = {
            "query": {
                "match": {
                    "表名": table_name
                }
            }
        }
    else:
        query = {"query": {"match_all": {}}}
    
    result = es.search(index=index, body=query, size=1000)
    return result['hits']['hits']

def main():
    while True:
        print("\n请选择操作：")
        print("1. 查询所有表")
        print("2. 搜索表（关键字搜索）")
        print("3. 查询表字段")
        print("4. 退出")
        
        choice = input("请输入选项（1-4）：")
        
        if choice == '1':
            results = search_tables()
            print("\n所有表信息：")
            for hit in results:
                print(hit['_source'])
                
        elif choice == '2':
            keyword = input("请输入搜索关键字：")
            results = search_tables(keyword)
            print(f"\n搜索结果：")
            for hit in results:
                print(hit['_source'])
                
        elif choice == '3':
            table_name = input("请输入表名（直接回车查看所有字段）：")
            results = search_fields(table_name)
            print(f"\n字段信息：")
            for hit in results:
                print(hit['_source'])
                
        elif choice == '4':
            print("退出程序")
            break
        else:
            print("无效的选项，请重新选择")

if __name__ == "__main__":
    main() 