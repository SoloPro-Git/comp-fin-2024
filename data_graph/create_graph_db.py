import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import networkx as nx
import json
from pathlib import Path
from data_graph.config import VALID_SOURCE_FIELDS, GENERIC_FIELDS, NUMERIC_KEYWORDS

def read_excel_data():
    df = pd.read_excel('data/数据字典/数据字典.xlsx', sheet_name='表字段信息')
    return df



def is_numeric_field(column_name):
    """判断是否是数值统计型字段"""
    column_name = str(column_name)
    return any(keyword in column_name for keyword in NUMERIC_KEYWORDS)

def get_numeric_fields(df):
    """获取所有数值统计型字段"""
    numeric_fields = []
    for col in df['column_description'].tolist():
        if is_numeric_field(col):
            numeric_fields.append(col)
    return numeric_fields

def is_valid_entity(column_name, numeric_fields, as_source=True):
    """判断一个字段是否应该作为实体
    
    Args:
        column_name: 字段名
        numeric_fields: 数值统计型字段列表
        as_source: 是否作为主实体（源节点）
    """
    # 检查是否是通用字段
    if column_name in GENERIC_FIELDS:
        return False
            
    # 如果是主实体，检查是否在允许的主实体列表中
    if as_source:
        return column_name in VALID_SOURCE_FIELDS
        
    # 如果是客实体，只要不是通用字段就可以
    return True

def create_graph(df):
    # 获取数值统计型字段
    numeric_fields = get_numeric_fields(df)
    print(f"\n找到 {len(numeric_fields)} 个数值统计型字段:")
    for field in numeric_fields:
        print(f"  - {field}")
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 按table_name分组处理数据
    grouped = df.groupby('table_name')
    
    # 为每个表创建实体和关系
    for table_name, group in grouped:
        # 获取该表的所有column_description
        columns = group['column_description'].tolist()
        
        # 为每个column创建节点（需要分别判断是否可以作为源节点和目标节点）
        for col in columns:
            if is_valid_entity(col, numeric_fields, as_source=True) or is_valid_entity(col, numeric_fields, as_source=False):
                if not G.has_node(col):
                    G.add_node(col, type='column')
        
        # 在同一表内的column之间创建关系
        for i, col1 in enumerate(columns):
            # 只有当col1可以作为源节点时才创建以它为起点的边
            if not is_valid_entity(col1, numeric_fields, as_source=True):
                continue
                
            for col2 in columns[i+1:]:
                # col2必须可以作为目标节点
                if not is_valid_entity(col2, numeric_fields, as_source=False):
                    continue
                    
                # 创建有向边（只从允许的主实体字段指向其他字段）
                G.add_edge(col1, col2, relation=table_name)
    
    return G

def save_graph(G, output_dir='data_graph/output'):
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 将图结构转换为字典格式
    graph_data = {
        'nodes': list(G.nodes()),
        'edges': [
            {
                'source': u,
                'target': v,
                'relation': data['relation']
            }
            for u, v, data in G.edges(data=True)
        ]
    }
    
    # 保存为JSON文件
    with open(f'{output_dir}/graph_data.json', 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    # 保存图的统计信息
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
    }
    
    with open(f'{output_dir}/graph_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def main():
    # 读取数据
    print("Reading Excel data...")
    df = read_excel_data()
    
    # 创建图
    print("Creating graph structure...")
    G = create_graph(df)
    
    # 保存图数据
    print("Saving graph data...")
    save_graph(G)
    
    print(f"Successfully created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges!")
    print("Data has been saved to the 'data_graph/output' directory.")

if __name__ == "__main__":
    main() 