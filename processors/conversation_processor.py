import sys,os
sys.path.append(os.getcwd())
import json
from llm.glm import GLM
from config import API_KEY
from processors.question_processor import QuestionProcessor

class ConversationProcessor:
    def __init__(self, model_name="glm-4-flash", output_dir='data/conversations'):
        # 初始化GLM模型
        self.glm = GLM(api_key=API_KEY, model=model_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 初始化问题处理器
        self.question_processor = QuestionProcessor(self.glm)

    def process_group(self, group):
        """处理单个问题组的对话"""
        tid = group['tid']
        print(f"\n开始处理对话组 {tid}")
        print("-" * 50)
        
        # 初始化对话历史
        messages = []
        
        # 设置system prompt
        messages = self.glm.set_system_prompt("你是一个helpful AI助手", messages)
        
        # 处理该组中的每个问题
        for qa in group['team']:
            messages = self.question_processor.process_question(qa, messages)
        
        # 保存对话记录
        self._save_conversation(tid, messages)
        print(f"\n对话已保存到 data/conversations/conversation_{tid}.json")
        
        return messages

    def _save_conversation(self, tid, messages):
        """保存对话记录"""
        output_file = os.path.join(self.output_dir, f'conversation_{tid}.json')
        
        # 提取对话内容
        conversation_log = []
        for msg in messages:
            if msg['role'] != 'system':  # 排除系统提示
                conversation_log.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_log, f, ensure_ascii=False, indent=2)

def test_processor():
    """用于测试的函数"""
    # 测试数据
    test_data = {
        "tid": "test-group-1",
        "team": [
            {
                "id": "tttt----1----1-1-1",
                "question": "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？"
            },
            {
                "id": "tttt----1----1-1-2",
                "question": "该公司实控人是否发生改变？如果发生变化，什么时候变成了谁？是哪国人？是否有永久境外居留权？（回答时间用XXXX-XX-XX）"
            },
            {
                "id": "tttt----1----1-1-3",
                "question": "在实控人发生变化的当年股权发生了几次转让？"
            }
        ]
    }
    
    try:
        # 初始化处理器
        processor = ConversationProcessor(model_name="glm-4-flash")
        
        # 处理测试数据
        print("开始测试处理器...")
        processor.process_group(test_data)
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    # 当直接运行此文件时，执行测试
    test_processor()