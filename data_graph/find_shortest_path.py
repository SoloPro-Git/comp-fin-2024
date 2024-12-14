import json
import networkx as nx
from pathlib import Path

def load_graph(graph_file='data_graph/output/graph_data.json'):
    """从JSON文件加载图数据并创建NetworkX图对象"""
    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # 添加节点
    for node in data['nodes']:
        G.add_node(node)
    
    # 添加边
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
    
    return G

def find_all_paths(G, start_node='公司代码', end_node='交易日', cutoff=3):
    """查找两个节点之间的所有路径（限制最大跳数为cutoff）"""
    try:
        # 使用NetworkX的所有简单路径算法
        all_paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=cutoff))
        
        # 按路径长度排序
        all_paths.sort(key=len)
        
        # 获取每条路径的详细信息
        paths_with_relations = []
        
        for path in all_paths:
            path_with_relations = []
            for i in range(len(path)-1):
                current = path[i]
                next_node = path[i+1]
                relation = G[current][next_node]['relation']
                path_with_relations.append({
                    'from': current,
                    'to': next_node,
                    'relation': relation
                })
            
            paths_with_relations.append({
                'path': path,
                'length': len(path) - 1,
                'detailed_path': path_with_relations
            })
        
        return {
            'total_paths_found': len(paths_with_relations),
            'paths': paths_with_relations
        }
    except nx.NodeNotFound as e:
        return {
            'error': f'节点不存在: {str(e)}'
        }

def main():
    # 加载图
    print("Loading graph data...")
    G = load_graph()
    
    # 查找所有路径
    print("\n查找从'公司代码'到'交易日'的所有路径...")
    result = find_all_paths(G)
    
    # 打印结果
    if 'error' in result:
        print(f"错误: {result['error']}")
    else:
        print(f"\n找到 {result['total_paths_found']} 条路径")
        print("\n所有路径详情:")
        for i, path_info in enumerate(result['paths'], 1):
            print(f"\n路径 {i} (长度: {path_info['length']} 跳):")
            for step in path_info['detailed_path']:
                print(f"  {step['from']} --[{step['relation']}]--> {step['to']}")
    
    # 保存结果到文件
    output_dir = Path('data_graph/output')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'all_paths_result.json'
    print(f"\n保存结果到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 