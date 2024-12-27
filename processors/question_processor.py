import json
import os
import sys
sys.path.append(os.getcwd())
from llm.glm import GLM
from data_graph.config import VALID_SOURCE_FIELDS
import pandas as pd
from data_graph.find_shortest_path import load_graph, find_all_paths
from utils.sql_request import SQLQueryClient
from utils.gen_full_schema_table_relation import generate_table_full_name_dict
import re

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
            glm_model: 主GLM模型实例
        """
        self.glm = glm_model
        self.table_full_name_dict = generate_table_full_name_dict()
        
        # 为实体提取创建专门的大模型实例
        self.entity_extraction_model = GLM(
            api_key=load_api_key(), 
            model="glm-4-air"
        )
        
        # 为SQL生成创建专门的大模型实例
        self.sql_generation_model = GLM(
            api_key=load_api_key(), 
            model="glm-4-air"
        )
        
        # 实体提取的系统提示词
        self.entity_extraction_system_prompt = """
        你是一个专业的金融数据实体提取专家，擅长从复杂的金融问题中精确提取关键实体信息。
        你的任务是识别问题中的具体实体字段和对应的值。

        工作原则：
        1. 仔细分析问题的语义和结构
        2. 准确识别问题中涉及的实体字段
        3. 提取实体字段对应的具体值
        4. 保持高度的准确性和一致性
        5. 如果无法确定，给出清晰的分析说明

        实体提取注意事项：
        - 关注金融领域的特定实体类型
        - 处理各种可能的表达方式
        - 区分不同类型的实体标识符
        - 保持对上下文的敏感性

        输出要求：
        1. 使用标准化的JSON格式
        2. 包含实体字段及其对应的值
        3. 提供简要的分析说明
        4. 确保输出的完整性和准确性
        """
        
        # SQL生成的系统提示词
        self.sql_generation_system_prompt = """
        你是一个专业的金融数据分析SQL专家，擅长根据复杂的问题和数据关系生成精确的SQL查询。
        你的任务是通过分析图路径、表关系和字段信息，为金融领域的复杂问题构建准确的SQL查询。

        工作原则：
        1. 仔细分析问题的具体信息需求
        2. 利用提供的图路径和表关系信息选择最合适的连接方式
        3. 确保SQL查询能精确回答问题
        4. 注意处理可能的多表关联和复杂条件
        5. 优先选择最短、最直接的查询路径
        6. 返回的SQL应当简洁、高效、易读

        SQL编写注意事项：
        - 使用标准MySQL语法
        - 优先使用LEFT JOIN
        - 在WHERE子句中添加必要的过滤条件
        - 使用明确的字段选择
        - 避免不必要的子查询
        - 考虑性能和可读性

        输出要求：
        1. 每个子问题对应一个精确的SQL查询
        2. 包含子问题描述
        3. 列出涉及的表
        4. 说明返回的字段
        5. 确保查询逻辑清晰且易于理解
        """
        
        self.graph_data = self._load_graph_data()
        self.valid_fields = self._load_valid_fields()
        # 初始化向量存储
        from utils.vector_store import VectorStore
        self.vector_store = VectorStore()
        # 加载数据字典
        self.data_dict = self._load_data_dictionary()
        # 加载图数据
        self.G = load_graph()
        self.sql_query_client = SQLQueryClient()

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

    def _extract_entities_from_question(self, question):
        """
        从问题中提取实体信息
        """
        # 准备消息列表并设置系统提示词
        messages = []
        messages = self.entity_extraction_model.set_system_prompt(
            self.entity_extraction_system_prompt, 
            messages
        )
        
        # 构建提示词（保持原有逻辑）
        prompt = f"""
        请分析以下问题中是否包含这些实体字段：
        实体字段列表：{', '.join(self.valid_fields)}

        问题：{question}
        实体字段的一些样例格式为：
        ###
            '公司名称': 'XXX实业',
            '证券代码': '6008XX',           # 公司唯一标识
            '证券代码': '',           # 证券唯一标识
            '证券内部编码': '',        # 证券内部唯一标识
            'JSID': '',             # 聚源数据标识
            'RID': '',              # 关系标识符
    
        ###

        请直接返回JSON格式数据（不要包含其他标记），格式如下：
        {{
            "analysis": "分析说明：解释为什么认为问题包含这些字段",
            "实体字段列表": {{"字段1": "实体1", "字段2": "实体2"}}
        }}
        """
        
        # 使用专门的实体提取模型
        reply, _ = self.entity_extraction_model.chat(prompt, messages)
        
        # 解析JSON（保持原有逻辑）
        try:
            result = self.parse_llm_json(reply, default_return={})
            
            # 处理实体字段列表（保持原有逻辑）
            entity_fields = result.get('实体字段列表', {})
            if isinstance(entity_fields, list):
                # 处理 ["字段1:实体1", "字段2:实体2"] 格式
                found_valid_entities = []
                for item in entity_fields:
                    if isinstance(item, str) and ':' in item:
                        field, entity = item.split(':', 1)
                        found_valid_entities.append({field.strip(): entity.strip()})
                    else:
                        found_valid_entities.append(item)
            elif isinstance(entity_fields, dict):
                # 处理 {"字段1": "实体1", "字段2": "实体2"} 格式
                found_valid_entities = [{k: v} for k, v in entity_fields.items()]
            else:
                found_valid_entities = []

            return {
                'has_valid_entities': bool(found_valid_entities),
                'found_entities': found_valid_entities,
                'nodes_exist': False,  # 暂时设为 False
                'found_nodes': [],     # 暂时为空列表
                'analysis': result.get('analysis', '')
            }
        except Exception as e:
            print(f"实体提取失败: {str(e)}")
            return {
                'has_valid_entities': False,
                'found_entities': [],
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
                        'table_details': row.get('表描述', '')
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

    def find_graph_paths_for_entities_and_terms(self, entities_analysis, relevant_terms, max_paths_per_term=5):
        """
        为实体和相关词语查找图中的路径
        
        Args:
            entities_analysis: 实体分析结果
            relevant_terms: 相关词语列表
            max_paths_per_term: 每个词语返回的最大路径数
        
        Returns:
            dict: 包含所有路径查找结果的字典
        """
        # 存储所有路径查找结果
        graph_paths_results = {}
        
        # 遍历每个主实体
        for main_entity_dict in entities_analysis['found_entities']:
            for main_entity_field, main_entity_value in main_entity_dict.items():
                print(f"\n主实体字段: {main_entity_field}")
                
                # 当为 ‘公司名称’ 时，替换为 ‘中文名称’
                if main_entity_field == '公司名称':
                    main_entity_field = '中文名称'
                
                # 为当前主实体创建结果列表
                graph_paths_results[main_entity_field] = {}
                
                # 遍历相关词语
                for term in relevant_terms:
                    try:
                        # 查找从主实体到相关词语的最短路径
                        path_result = find_all_paths(self.G, start_node=main_entity_field, end_node=term['text'], cutoff=5)
                        
                        print(f"\n从 {main_entity_field} 到 {term['text']} 的路径:")
                        if 'error' in path_result:
                            print(f"错误: {path_result['error']}")
                            graph_paths_results[main_entity_field][term['text']] = {'error': path_result['error']}
                        else:
                            # 按路径长度排序
                            sorted_paths = sorted(path_result['paths'], key=lambda x: x['length'])
                            
                            # 限制返回的路径数量
                            limited_paths = sorted_paths[:max_paths_per_term]
                            
                            print(f"找到 {path_result['total_paths_found']} 条路径，返回最短的 {len(limited_paths)} 条")
                            
                            # 存储路径详情
                            graph_paths_results[main_entity_field][term['text']] = {
                                'total_paths_found': path_result['total_paths_found'],
                                'paths': []
                            }
                            
                            for i, path_info in enumerate(limited_paths, 1):
                                print(f"\n路径 {i} (长度: {path_info['length']} 跳):")
                                detailed_path = []
                                for step in path_info['detailed_path']:
                                    print(f"  {step['from']} --[{step['relation']}]--> {step['to']}")
                                    detailed_path.append({
                                        'from': step['from'],
                                        'to': step['to'],
                                        'relation': step['relation']
                                    })
                                
                                graph_paths_results[main_entity_field][term['text']]['paths'].append({
                                    'length': path_info['length'],
                                    'detailed_path': detailed_path
                                })
                
                    except Exception as e:
                        print(f"查找路径时发生错误: {str(e)}")
                        graph_paths_results[main_entity_field][term['text']] = {'error': str(e)}
        
        return graph_paths_results

    def extract_graph_paths_details(self, graph_paths):
        """
        提取图路径的详细信息和文本结果
        
        Args:
            graph_paths: find_graph_paths_for_entities_and_terms 返回的路径结果
        
        Returns:
            dict: 包含路径结果字符串、涉及的表名和详细信息
        """
        # 存储所有涉及的节点和关系（可能的表名）
        all_nodes = set()
        all_relations = set()
        
        # 存储路径结果的字符串表示
        paths_str = []
        
        # 存储路径的详细信息（用于传递给LLM）
        paths_details = []
        
        # 遍历路径结果
        for main_entity, term_paths in graph_paths.items():
            main_entity_paths = []
            paths_str.append(f"主实体: {main_entity}")
            
            for term, path_info in term_paths.items():
                term_entity_paths = {
                    'target_term': term,
                    'paths': []
                }
                
                paths_str.append(f"  目标词语: {term}")
                
                if 'error' in path_info:
                    paths_str.append(f"    错误: {path_info['error']}")
                    term_entity_paths['error'] = path_info['error']
                else:
                    paths_str.append(f"    找到 {path_info['total_paths_found']} 条路径")
                    
                    for i, path in enumerate(path_info['paths'], 1):
                        path_str = f"    路径 {i} (长度: {path['length']} 跳):"
                        paths_str.append(path_str)
                        
                        detailed_path = []
                        path_steps = []
                        for step in path['detailed_path']:
                            step_str = f"      {step['from']} --[{step['relation']}]--> {step['to']}"
                            paths_str.append(step_str)
                            
                            # 收集所有节点和关系
                            all_nodes.add(step['from'])
                            all_nodes.add(step['to'])
                            all_relations.add(step['relation'])
                            
                            # 记录路径步骤
                            path_steps.append({
                                'from': step['from'],
                                'to': step['to'],
                                'relation': step['relation']
                            })
                        
                        term_entity_paths['paths'].append({
                            'length': path['length'],
                            'steps': path_steps
                        })
                    
                    main_entity_paths.append(term_entity_paths)
            
            paths_details.append({
                'main_entity': main_entity,
                'term_paths': main_entity_paths
            })
        
        # 移除可能的空节点和关系
        all_nodes = {node for node in all_nodes if node}
        all_relations = {relation for relation in all_relations if relation}
        
        # 使用 find_table_details_from_table_names 查找表详细信息
        # 将 all_relations 转换为唯一的表名列表
        unique_table_names = list(all_relations)
        
        # 查找表的详细信息
        table_details = self.find_table_details_from_table_names(unique_table_names)
        
        # 打印表详情
        print("\n路径中涉及的表详细信息:")
        for node, detail in table_details.items():
            print(f"节点: {node}")
            print(f"  - 表名(中文): {detail['table_name_cn']}")
            print(f"  - 表名(英文): {detail['table_name_en']}")
            print(f"  - 所属数据库(中文): {detail['database_cn']}")
            print(f"  - 所属数据库(英文): {detail['database_en']}")
            print("---")
        
        return {
            'paths_str': '\n'.join(paths_str),
            'paths_details': paths_details,
            'table_details': table_details
        }

    def find_table_columns_info(self, unique_table_names):
        """
        在"表字段信息"sheet中查找指定表的字段信息
        
        Args:
            unique_table_names: 要查找的表名列表
        
        Returns:
            dict: 表名对应的字段信息
        """
        # 存储表的字段信息
        table_columns_info = {}
        
        # 确保存在"表字段信息"sheet
        if '表字段信息' not in self.data_dict:
            print("未找到表字段信息sheet")
            return table_columns_info
        
        # 获取"表字段信息"sheet的DataFrame
        table_fields_df = self.data_dict['表字段信息']['df']
        
        # 遍历每个表名
        for table_name in unique_table_names:
            # 在表字段信息中查找匹配的行
            matching_rows = table_fields_df[
                table_fields_df['table_name'].astype(str) == table_name
            ]
            
            # 如果找到匹配行
            if not matching_rows.empty:
                # 提取字段信息
                columns_info = []
                for _, row in matching_rows.iterrows():
                    column_info = {
                        'column_name': row.get('column_name', ''),
                        'column_description': row.get('column_description', '')
                    }
                    columns_info.append(column_info)
                
                # 存储该表的字段信息
                table_columns_info[table_name] = columns_info
        
        # 打印表字段信息
        print("\n表字段详细信息:")
        for table, columns in table_columns_info.items():
            print(f"表名: {table}")
            for column in columns:
                print(f"  - 字段名: {column['column_name']}")
                print(f"    描述: {column['column_description']}")
            print("---")
        
        return table_columns_info

    def parse_llm_json(self, reply, default_return={"queries": []}):
        """
        解析大模型返回的 JSON 字符串
        
        Args:
            reply: 大模型返回的文本
            default_return: 解析失败时返回的默认值
        
        Returns:
            解析后的 JSON 对象
        """
        try:
            # 清理和预处理回复文本
            reply = reply.strip()
            
            # 移除代码块标记
            if reply.startswith('```json'):
                reply = reply[7:-3]
            elif reply.startswith('```'):
                reply = reply[3:-3]
            
            # 尝试解析 JSON
            parsed_json = json.loads(reply)
            
            return parsed_json
        
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            print(f"原始响应: {reply}")
            
            # 尝试处理可能的特殊情况
            try:
                # 移除可能的前导/后导文本
                start_index = reply.find('{')
                end_index = reply.rfind('}') + 1
                
                if start_index != -1 and end_index != -1:
                    cleaned_reply = reply[start_index:end_index]
                    parsed_json = json.loads(cleaned_reply)
                    
                    print("\n清理后解析的JSON:")
                    print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
                    
                    return parsed_json
            except Exception as clean_error:
                print(f"清理JSON时发生错误: {str(clean_error)}")
            
            return default_return
        except Exception as e:
            print(f"解析过程中发生错误: {str(e)}")
            return default_return

    def generate_sql_queries_from_paths(self, question, processed_paths):
        """
        根据图路径、表信息和字段信息，生成用于回答问题的SQL查询
        """
        # 准备消息列表并设置系统提示词
        messages = []
        messages = self.sql_generation_model.set_system_prompt(
            self.sql_generation_system_prompt, 
            messages
        )
        
        # 后续代码保持原有逻辑
        paths_details_prompt = json.dumps(processed_paths['paths_details'], ensure_ascii=False, indent=2)
        table_details_prompt = json.dumps(processed_paths['table_details'], ensure_ascii=False, indent=2)
        table_columns_prompt = json.dumps(
            self.find_table_columns_info(list(processed_paths['table_details'].keys())), 
            ensure_ascii=False, 
            indent=2
        )
        
        # 构建LLM提示词
        llm_prompt = f"""
        任务：根据图路径、表信息和字段信息，为以下问题生成精确的SQL查询语句

        原始问题：{question}

        图路径详情：
        {paths_details_prompt}

        涉及表详情：
        {table_details_prompt}

        表字段信息：
        {table_columns_prompt}

        请按照以下要求完成任务：
        1. 仔细分析问题中需要查询的具体信息
        2. 根据图路径和表信息，选择合适的表和字段
        3. 生成能够精确回答问题的SQL查询
        4. 使用JSON格式返回，包含：
        - 子问题描述
        - 对应的SQL查询
        - 涉及的表
        - 预期返回的字段

        子问题需要根据原始问题进行分解，图路径、表信息仅作参考。
        
        查表优先选择SecuMain、LC_AreaCode这些表。
        原始问题提及港股才使用HK_XXX的表，提及美股才使用US_XXX的表，否则不使用HK_XXX、US_XXX的表。
        
        返回格式示例：
        {{
            "queries": [
                {{
                    "sub_question": "查询公司全称",
                    "sql": "SELECT 表名.全称字段 FROM 表名 WHERE 公司代码 = 'xxx'",
                    "involved_tables": ["表名1", "表名2"],
                    "return_fields": ["全称", "公司代码"]
                }},
                // 其他子查询...
            ]
        }}
        """
        
        try:
            # 使用专门的SQL生成模型
            reply, _ = self.sql_generation_model.chat(llm_prompt, messages)
            
            # 解析JSON
            sql_queries = self.parse_llm_json(reply)
            
            # 打印生成的SQL查询
            print("\n生成的SQL查询:")
            for query in sql_queries.get('queries', []):
                print(f"子问题: {query.get('sub_question', '')}")
                print(f"SQL: {query.get('sql', '')}")
                print(f"涉及表: {query.get('involved_tables', [])}")
                print(f"返回字段: {query.get('return_fields', [])}")
                print("---")
            
            return sql_queries
        
        except Exception as e:
            print(f"SQL查询生成失败: {str(e)}")
            return {"queries": []}

    def generate_final_answer(self, question, sql_queries, messages):
        """
        根据SQL查询结果生成最终答案
        
        Args:
            question: 原始问题
            sql_queries: 生成的SQL查询
            messages: 对话历史
        
        Returns:
            最终答案和更新后的消息历史
        """
        # 准备存储查询结果
        query_results = []
        
        # 执行每个SQL查询
        for query in sql_queries.get('queries', []):
            sql = query.get('sql', '')
            sub_question = query.get('sub_question', '')
            
            try:
                # 使用 SQLQueryClient 执行查询
                result = self.sql_query_client.execute_query(sql)
                
                query_results.append({
                    'sub_question': sub_question,
                    'sql': sql,
                    'result': result
                })
            except Exception as e:
                print(f"执行查询时发生错误: {str(e)}")
                query_results.append({
                    'sub_question': sub_question,
                    'sql': sql,
                    'error': str(e)
                })
        
        # 准备提示词，让 LLM 根据查询结果生成答案
        answer_generation_prompt = f"""
        原始问题: {question}

        SQL查询结果:
        {json.dumps(query_results, ensure_ascii=False, indent=2)}

        请根据上述查询结果，生成一个清晰、准确的回答。
        要求:
        1. 直接回答原始问题
        2. 如果某些子查询出错，说明无法获取该部分信息
        3. 保持答案的专业性和简洁性
        """
        
        # 使用 GLM 模型生成最终答案
        final_answer, updated_messages = self.glm.chat(answer_generation_prompt, messages)
        
        return final_answer, updated_messages

    def replace_table_names_with_full_names(self, sql_queries):
        """
        将表名替换为完整的 schema.table 格式
        
        Args:
            sql_queries (dict): 包含查询信息的字典
        
        Returns:
            dict: 更新后的查询信息字典
        """
        for query in sql_queries['queries']:
            updated_tables = []
            
            # 替换 involved_tables
            for table in query['involved_tables']:
                full_table_name = self.table_full_name_dict.get(table, table)
                updated_tables.append(full_table_name)
            query['involved_tables'] = updated_tables
            
            # 替换 SQL 查询中的表名
            original_sql = query.get('sql', '')
            updated_sql = original_sql
            
            for short_table_name, full_table_name in self.table_full_name_dict.items():
                # 使用正则表达式替换，确保只替换完整的表名
                updated_sql = re.sub(r'\b{}\b'.format(re.escape(short_table_name)), full_table_name, updated_sql)
            
            query['sql'] = updated_sql
        
        return sql_queries

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
        
        # 获取问题中可能匹配的客体
        relevant_terms = self.get_relevant_terms(question)
               
        # 提取问题中的实体信息
        entities_analysis = self._extract_entities_from_question(question)
        print("\n实体提取结果:")
        print(f"包含有效实体: {entities_analysis['has_valid_entities']}")
        print(f"找到的有效实体: {entities_analysis['found_entities']}")
        
        # 查找图中的路径
        graph_paths = self.find_graph_paths_for_entities_and_terms(entities_analysis, relevant_terms)
        
        # 提取图路径详细信息
        processed_paths = self.extract_graph_paths_details(graph_paths)
                
        # 生成SQL查询
        sql_queries = self.generate_sql_queries_from_paths(question, processed_paths)

        # 更新表名为schema.table格式
        sql_queries = self.replace_table_names_with_full_names(sql_queries)
        
        # 根据SQL查询结果生成最终答案
        final_answer, messages = self.generate_final_answer(question, sql_queries, messages)
        
        print("\n最终答案:")
        print(final_answer)
        
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
            # 使用hybrid_search进行路召回
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
            print(f"获取相关词语时生错误: {str(e)}")
            return []

def test_question_processor():
    """测试问题处理器"""
    GLM_API_KEY = load_api_key()
    
    # 测试数据
    test_qa = {
        "id": "tttt----1----1-1-1",
        "question": "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？"
    }
    
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

if __name__ == "__main__":
    test_question_processor()
