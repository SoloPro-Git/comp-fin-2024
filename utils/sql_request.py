import os,sys
sys.path.append(os.getcwd())
import requests
import json
try:
    with open('apikey', 'r') as f:
        for line in f:
            if line.startswith('COMPETITION_API_KEY='):
                COMPETITION_API_KEY = line.strip().split('=')[1]
                break
        else:
            raise ValueError("未找到COMPETITION_API_KEY配置")
except FileNotFoundError:
    raise FileNotFoundError("未找到apikey文件，请在根目录创建apikey文件")
class SQLQueryClient:
    def __init__(self, api_key=None):
        """
        初始化 SQL 查询客户端
        
        :param api_key: API 密钥，如果未提供，则尝试从环境变量读取
        """
        self.api_key = COMPETITION_API_KEY
        if not self.api_key:
            raise ValueError("未提供 API Key，请设置 COMPETITION_API_KEY 环境变量")
        
        self.url = "https://comm.chatglm.cn/finglm2/api/query"
        self.databases_url = "https://comm.chatglm.cn/finglm2/api/databases"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def get_databases(self):
        """
        获取所有可用的数据库列表
        
        :return: 数据库列表或 None
        """
        try:
            response = requests.get(self.databases_url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取数据库列表失败：{e}")
            return None

    def execute_query(self, sql_query, limit=10):
        """
        执行 SQL 查询
        
        :param sql_query: SQL 查询语句
        :param limit: 返回结果的最大行数，默认为 10
        :return: 查询结果的 JSON 数据
        """
        data = {
            "sql": sql_query,
            "limit": limit
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()  # 检查请求是否成功
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"查询请求失败：{response.json()['detail']}")
            return None

    def pretty_print_result(self, result):
        """
        格式化打印查询结果
        
        :param result: 查询结果的 JSON 数据
        """
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("没有查询结果")

# 使用示例
if __name__ == "__main__":
    client = SQLQueryClient()
    
    # 获取数据库列表
    # print("可用数据库列表：")
    # databases = client.get_databases()
    # client.pretty_print_result(databases)
    
    # 示例查询
    query = "SELECT ChiName FROM ConstantDB.SecuMain WHERE SecuCode = '600872'"
    result = client.execute_query(query)
    client.pretty_print_result(result)