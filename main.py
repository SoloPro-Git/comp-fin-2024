import json
from processors.conversation_processor import ConversationProcessor

def load_questions(file_path):
    """加载问题文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 初始化对话处理器
    processor = ConversationProcessor(model_name="glm-4")
    
    # 加载问题
    question_groups = load_questions('data/question.json')
    
    # 处理每组对话
    for group in question_groups:
        processor.process_group(group)

if __name__ == "__main__":
    main() 