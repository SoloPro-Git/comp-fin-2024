import json
import os
import sys
sys.path.append(os.getcwd())
from llm.glm import GLM
from data_graph.config import VALID_SOURCE_FIELDS

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
        
        # 构造提示词
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

    def process_question(self, qa, messages):
        """
        处理单个问题
        Args:
            qa: 包含问题ID和内容的字典
            messages: 当前的对话历史
        Returns:
            更新后的对话历史
        """
        question_id = qa['id']
        question = qa['question']
        
        print(f"\n问题ID: {question_id}")
        print(f"用户: {question}")
        
        # 分析问题
        analysis = self._analyze_question(question)
        print("\n问题分析结果:")
        print(f"包含有效字段: {analysis['has_valid_fields']}")
        print(f"找到的有效字段: {analysis['found_fields']}")
        print(f"包含图数据字段: {analysis['nodes_exist']}")
        print(f"找到的图数据节点: {analysis['found_nodes']}")
        print(f"分析说明: {analysis['analysis']}")
        
        # 发送问题并获取回复
        # reply, messages = self.glm.chat(question, messages)
        # print(f"AI: {reply}")
        
        return messages

def test_question_processor():
    """测试问题处理器"""
    from config import API_KEY
    
    # 测试数据
    test_qa = {
        "id": "tttt----1----1-1-1",
        "question": "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？"
    }
    
    # try:
        # 初始化GLM模型
    glm = GLM(api_key=API_KEY, model="glm-4-air")
    
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