# 测试：是否能连上通义千问AI
import requests

API_KEY = "sk-84090b26de1442dfa935a8d3b25d9cc2"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def test_ai():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": "你好"}]
    }

    print("正在调用 AI...")
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        print("状态码:", response.status_code)
        print("返回结果:", response.text)  # 打印全部真实返回！
    except Exception as e:
        print("连接失败原因:", e)

test_ai()