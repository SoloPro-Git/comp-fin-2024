import json
import os
import sys
sys.path.append(os.getcwd())
from llm.glm import GLM
from data_graph.config import VALID_SOURCE_FIELDS
import pandas as pd

def load_api_key():
    """从apikey文件加载API密钥"""
    try:
        with open('apikey', 'r') as f:
            for line in f:
                if line.startswith('GLM_API_KEY='):
                    return line.strip().split('=')[1]
        raise ValueError("未找到GLM_API_KEY配置")
    except FileNotFoundError:
        raise FileNotFoundError("未找到apikey文件，请在根目录创建apikey文件")

class QuestionProcessor:
    def __init__(self, glm_model):
        """
        初始化问题处理器
        Args:
            glm_model: GLM模型实例
        """
        self.glm = glm_model
        self.graph_data = self._load_graph_data()
        self.valid_fields = self._load_valid_fields()
        # 初始化向量存储
        from utils.vector_store import VectorStore
        self.vector_store = VectorStore()
        # 加载数据字典
        self.data_dict = self._load_data_dictionary()

    def _load_graph_data(self):
        """加载图数据"""
        graph_file = "data_graph/output/graph_data.json"
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"找不到图数据文件: {graph_file}")
        
        with open(graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_valid_fields(self):
        """加载有效字段列表"""
        if not VALID_SOURCE_FIELDS:
            raise ValueError("VALID_SOURCE_FIELDS 未在 data_graph/config.py 中定义")
        return VALID_SOURCE_FIELDS

    def _load_data_dictionary(self):
        """
        加载数据字典Excel文件
        Returns:
            dict: 包含表名和字段信息的字典
        """
        try:
            # 读取Excel文件
            excel_path = "data/数据字典/数据字典.xlsx"
            
            # 读取所有sheet
            xls = pd.ExcelFile(excel_path)
            data_dict = {}
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                data_dict[sheet_name] = {
                    'fields': df.columns,
                    'df': df  # 保存完整的DataFrame以便后续使用
                }
            
            return data_dict
        
        except Exception as e:
            print(f"加载数据字典时发生错误: {str(e)}")
            return {}

    def _analyze_question(self, question):
        """
        分析问题中的关键字段
        Returns:
            dict: {
                'has_valid_fields': bool,  # 是否包含有效字段
                'found_fields': list,      # 找到的有效字段列表
                'nodes_exist': bool,       # 字段是否存在于图数据中
                'found_nodes': list        # 在图数据中找到的节点
            }
        """
        # 获取所有节点的字段名称
        node_fields = set()
        for node in self.graph_data.get('nodes', []):
            node_name = str(node)
            if node_name:
                node_fields.add(node_name)
        
        # 构提示词
        prompt = f"""
        请分析以下问题中是否包含这些字段：
        1. 实体字段列表：{', '.join(self.valid_fields)}
        2. 查询数据字段：{', '.join(node_fields)}

        问题：{question}
        
        请直接返回JSON格式数据（不要包含其他标记），格式如下：
        {{
            "analysis": "分析说明：解释为什么认为问题包含这些字段",
            "实体字段列表": {{"字段1": "实体1", "字段2": "实体2"}},
            "查询数据字段": ["字段1", "字段2"]
        }}
        """
        
        # 获取模型回答
        reply, _ = self.glm.chat(prompt, [])
        
        try:
            # 清理回答中的可能干扰 JSON 解析的内容
            reply = reply.strip()
            if reply.startswith('```'):
                reply = reply[reply.find('{'):reply.rfind('}')+1]
            
            # 解析模型回答
            result = json.loads(reply)
            
            # 处理实体字段列表（兼容两种格式）
            entity_fields = result.get('实体字段列表', {})
            if isinstance(entity_fields, list):
                # 处理 ["字段1:实体1", "字段2:实体2"] 格式
                found_valid_fields = []
                for item in entity_fields:
                    if isinstance(item, str) and ':' in item:
                        field, entity = item.split(':', 1)
                        found_valid_fields.append({field.strip(): entity.strip()})
                    else:
                        found_valid_fields.append(item)
            elif isinstance(entity_fields, dict):
                # 处理 {"字段1": "实体1", "字段2": "实体2"} 格式
                found_valid_fields = [{k: v} for k, v in entity_fields.items()]
            else:
                found_valid_fields = []

            # 处理查询数据字段
            found_node_fields = result.get('查询数据字段', [])
            
            # 检查字段是否存在于图数据中
            confirmed_nodes = [node for node in found_node_fields if node in node_fields]
            
            return {
                'has_valid_fields': bool(found_valid_fields),
                'found_fields': found_valid_fields,
                'nodes_exist': bool(confirmed_nodes),
                'found_nodes': confirmed_nodes,
                'analysis': result.get('analysis', '')
            }
        except json.JSONDecodeError as e:
            print(f"警告：模型返回的JSON格式无效: {reply}")
            print(f"错误详情: {str(e)}")
            return {
                'has_valid_fields': False,
                'found_fields': [],
                'nodes_exist': False,
                'found_nodes': [],
                'analysis': ''
            }

    def find_table_names_for_terms(self, relevant_terms):
        """
        根据相关词语查找对应的表名
        Args:
            relevant_terms: 相关词语列表
        Returns:
            dict: 词语对应的表名及详细信息
        """
        table_matches = {}
        
        for term in relevant_terms:
            text = term['text']
            matches = []
            
            # 遍历所有sheet和字段
            for sheet_name, sheet_info in self.data_dict.items():
                df = sheet_info['df']
                
                # 检查是否有 column_description 和 table_name 列
                if 'column_description' in df.columns and 'table_name' in df.columns:
                    # 查找 column_description 列中包含该词语的行
                    matching_rows = df[df['column_description'].astype(str).str.contains(text, na=False)]
                    
                    if not matching_rows.empty:
                        # 获取匹配行的 table_name 值
                        table_names = matching_rows['table_name'].unique().tolist()
                        
                        matches.append({
                            'sheet_name': sheet_name,
                            'matching_table_names': table_names,
                            'matching_descriptions': matching_rows['column_description'].tolist()
                        })
                
                # 备选方案：如果没有完全匹配的列名
                elif 'column' in df.columns and 'table' in df.columns:
                    matching_rows = df[df['column'].astype(str).str.contains(text, na=False)]
                    
                    if not matching_rows.empty:
                        table_names = matching_rows['table'].unique().tolist()
                        
                        matches.append({
                            'sheet_name': sheet_name,
                            'matching_table_names': table_names,
                            'matching_descriptions': matching_rows['column'].tolist()
                        })
                
                # 如果找到匹配，存储
                if matches:
                    table_matches[text] = matches
            
            # 打印匹配结果
            print("\n表名匹配结果:")
            for term, match_info in table_matches.items():
                print(f"词语: {term}")
                for match in match_info:
                    print(f"  - 表名: {match['sheet_name']}")
                    print(f"  - 匹配的 table_name: {match['matching_table_names']}")
                    print(f"  - 匹配的描述: {match['matching_descriptions']}")
        
        return table_matches

    def find_table_details_from_table_names(self, unique_table_names):
        """
        根据表名在库表关系sheet中查找详细信息
        Args:
            unique_table_names: 唯一的表名列表
        Returns:
            dict: 表的详细信息
        """
        table_details = {}
        
        # 确保存在库表关系sheet
        if '库表关系' not in self.data_dict:
            print("未找到库表关系sheet")
            return table_details
        
        # 获取库表关系sheet的DataFrame
        library_relations_df = self.data_dict['库表关系']['df']
        
        # 遍历每个表名
        for table_name in unique_table_names:
            # 在库表关系中查找匹配的行
            matching_rows = library_relations_df[
                (library_relations_df['表中文'] == table_name) | 
                (library_relations_df['表英文'] == table_name)
            ]
            
            # 如果找到匹配行
            if not matching_rows.empty:
                for _, row in matching_rows.iterrows():
                    row_details = {
                        'table_name': table_name,
                        'table_name_cn': row.get('表中文', ''),
                        'table_name_en': row.get('表英文', ''),
                        'database_en': row.get('库名英文', ''),
                        'database_cn': row.get('库名中文', ''),
                    }
                    table_details[table_name] = row_details
        
        # 打印表详情
        print("\n表详细信息:")
        for table_name, detail in table_details.items():
            print(f"表名: {table_name}")
            print(f"  - 表名(中文): {detail['table_name_cn']}")
            print(f"  - 表名(英文): {detail['table_name_en']}")
            print(f"  - 所属数据库(中文): {detail['database_cn']}")
            print(f"  - 所属数据库(英文): {detail['database_en']}")
            print("---")
        
        return table_details

    def extract_table_names_from_matches(self, table_matches):
        """
        从 table_matches 中提取所有唯一的表名
        Args:
            table_matches: find_table_names_for_terms 返回的匹配结果
        Returns:
            list: 所有唯一的表名
        """
        all_table_names = set()
        
        for term, matches in table_matches.items():
            for match in matches:
                # 从 matching_table_names 中提取表名
                table_names = match.get('matching_table_names', [])
                all_table_names.update(table_names)
        
        # 转换为列表并打印
        unique_table_names = list(all_table_names)
        
        print("\n提取的表名:")
        for name in unique_table_names:
            print(f"  - {name}")
        
        return unique_table_names

    def process_question(self, qa, messages):
        """
        处理单个问题
        Args:
            qa: 包含题ID和内容的字典
            messages: 当前的对话历史
        Returns:
            更新后的对话历史
        """
        question_id = qa['id']
        question = qa['question']
        
        print(f"\n问题ID: {question_id}")
        print(f"用户: {question}")
        
        # 新增：获取相关词语
        relevant_terms = self.get_relevant_terms(question)
        
        # 查找相关词语对应的表名
        table_matches = self.find_table_names_for_terms(relevant_terms)
        
        # 提取所有表名
        unique_table_names = self.extract_table_names_from_matches(table_matches)
        
        # 查找表的详细信息
        table_details = self.find_table_details_from_table_names(unique_table_names)
        
        # 分析问题
        analysis = self._analyze_question(question)
        print("\n问题分析结果:")
        print(f"包含有效字段: {analysis['has_valid_fields']}")
        print(f"找到的有效字段: {analysis['found_fields']}")
        print(f"包含图数据字段: {analysis['nodes_exist']}")
        print(f"找到的图数据节点: {analysis['found_nodes']}")
        print(f"分析说明: {analysis['analysis']}")
        
        return messages

    def get_relevant_terms(self, question):
        """
        使用混合搜索获取与问题相关的词语
        Args:
            question: 用户提出的问题
        Returns:
            list: 相关词语列表及其得分
        """
        try:
            # 使用hybrid_search进行多路召回
            search_results = self.vector_store.hybrid_search(question, top_k=5)
            
            # 格式化搜索结果
            relevant_terms = []
            for result in search_results:
                relevant_terms.append({
                    'text': result['text'],
                    'relevance_score': result['score'],
                    'vector_score': result['vector_score'],
                    'bm25_score': result['bm25_score']
                })
                
            print("\n多路召回结果:")
            for term in relevant_terms:
                print(f"文本: {term['text']}")
                print(f"相关度得分: {term['relevance_score']:.4f}")
                print(f"向量相似度: {term['vector_score']:.4f}")
                print(f"BM25得分: {term['bm25_score']:.4f}")
                print("---")
                
            return relevant_terms
            
        except Exception as e:
            print(f"获取相关词语时发生错误: {str(e)}")
            return []

def test_question_processor():
    """测试问题处理器"""
    GLM_API_KEY = load_api_key()
    
    # 测试数据
    test_qa = {
        "id": "tttt----1----1-1-1",
        "question": "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？"
    }
    
    # try:
        # 初始化GLM模型
    glm = GLM(api_key=GLM_API_KEY, model="glm-4-air")
    
    # 初始化问题处理器
    processor = QuestionProcessor(glm)
    
    # 测试处理单个问题
    print("开始测试问题处理器...")
    messages = []
    messages = glm.set_system_prompt("你是一个helpful AI助手", messages)
    messages = processor.process_question(test_qa, messages)
    print("\n测试完成！")
        
    # except Exception as e:
    #     print(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    test_question_processor() 