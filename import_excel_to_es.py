import pandas as pd
from elasticsearch import Elasticsearch
import json

# 创建ES客户端连接
es = Elasticsearch(
    ['http://localhost:9200'],
    basic_auth=('elastic', 'elastic')  # 添加认证信息
)

# 读取Excel文件
def read_excel_sheets(file_path):
    sheets_data = {}
    
    # 指定要读取的sheet名称
    target_sheets = ['库表关系', '表字段信息']
    
    for sheet_name in target_sheets:
        # 读取sheet内容
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 如果存在 column_description 列，进行替换
        if 'column_description' in df.columns:
            # 定义替换规则
            replace_dict = {
                '中文名称': '公司名称',
                '中文简称': '公司简称',
                '中文名称缩写': '公司名称缩写',
                '英文名称': '英文公司名称',
                '英文名称缩写': '英文公司名称缩写'
            }
            df['column_description'] = df['column_description'].replace(replace_dict)
            print("替换后的唯一值:", df['column_description'].unique())  # 添加这行来检查
        
        # 将DataFrame转换为字典列表
        records = df.to_dict('records')
        sheets_data[sheet_name] = records
    
    return sheets_data

def import_to_es(sheets_data):
    # 定义索引映射关系
    sheet_to_index = {
        '库表关系': '库表关系',
        '表字段信息': '表字段信息'
    }
    
    for sheet_name, records in sheets_data.items():
        # 使用预定义的索引名
        index_name = sheet_to_index[sheet_name]
        
        # 检查索引是否存在，如果存在则删除
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
        
        # 创建新索引
        es.indices.create(index=index_name)
        
        # 准备批量导入的数据
        bulk_data = []
        for i, record in enumerate(records):
            # 清理数据中的NaN值
            clean_record = {k: ('' if pd.isna(v) else v) for k, v in record.items()}
            
            # 添加索引操作指令
            bulk_data.append({
                "index": {
                    "_index": index_name,
                    "_id": i
                }
            })
            # 添加文档数据
            bulk_data.append(clean_record)
        
        if bulk_data:
            # 执行批量导入
            es.bulk(operations=bulk_data)
            print(f"已成功批量导入 {len(records)} 条记录到索引 {index_name}")

def main():
    try:
        # Excel文件路径
        file_path = "data/数据字典/数据字典.xlsx"
        
        # 读取Excel数据
        print("正在读取Excel文件...")
        sheets_data = read_excel_sheets(file_path)
        
        # 导入到ES
        print("正在导入数据到Elasticsearch...")
        import_to_es(sheets_data)
        
        print("所有数据导入完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 