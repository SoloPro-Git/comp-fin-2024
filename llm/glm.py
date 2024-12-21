import os, sys
sys.path.append(os.getcwd())
from zhipuai import ZhipuAI
from jinja2 import Template
try:
    with open('apikey', 'r') as f:
        for line in f:
            if line.startswith('GLM_API_KEY='):
                GLM_API_KEY = line.strip().split('=')[1]
                break
        else:
            raise ValueError("未找到GLM_API_KEY配置")
except FileNotFoundError:
    raise FileNotFoundError("未找到apikey文件，请在根目录创建apikey文件")

class GLM:
    def __init__(self, api_key='None', base_url=None, model='glm-4'):
        self.model = model
        if base_url:
            self.client = ZhipuAI(api_key=api_key, base_url=base_url)
        else:
            self.client = ZhipuAI(api_key=api_key)

    def set_system_prompt(self, system_prompt, messages):
        """设置 system prompt"""
        messages.insert(0, {"role": "system", "content": system_prompt})
        return messages
    
    def set_query_template(self, query_template, **kwargs):
        """使用模板设置查询"""
        template = Template(query_template)
        query_with_template = template.render(**kwargs)
        return query_with_template

    def chat(self, query, messages=[]):
        """支持多轮对话"""
        if not messages:
            messages = [{"role": "user", "content": query}]
        else:
            messages.append({"role": "user", "content": query})
        
        # 过滤空内容
        messages = [i for i in messages if i['content']]
        
        # 调用 API 获取回复
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
        )
        
        # 获取回复内容
        reply = response.choices[0].message.content
        
        # 将助手的回复添加到对话历史
        messages.append({"role": "assistant", "content": reply})
        
        return reply, messages

if __name__ == "__main__":
    # 示例用法
    glm = GLM(api_key=GLM_API_KEY,model="glm-4-flash")
    
    # 设置 system prompt
    messages = []
    messages = glm.set_system_prompt("你是一个helpful AI助手", messages)
    
    # 第一轮对话
    reply, messages = glm.chat("你好", messages)
    print("AI:", reply)
    
    # 第二轮对话
    reply, messages = glm.chat("今天天气怎么样？", messages)
    print("AI:", reply)