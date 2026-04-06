# 09_requests_demo.py
# Python requests 库操作

import requests

# 注意：运行此文件前需要先安装 requests 库
# 安装命令：pip install requests

print("=== requests 库基础操作 ===\n")

# 1. GET 请求
print("1. GET 请求")

# 使用 httpbin.org 测试 API
response = requests.get("https://httpbin.org/get", params={"name": "Alice", "age": 25})

print(f"状态码：{response.status_code}")
print(f"请求的 URL: {response.url}")

# 解析响应
data = response.json()
print(f"请求参数：{data['args']}")

# 2. POST 请求
print("\n2. POST 请求")

post_data = {
    "name": "Bob",
    "age": 30,
    "city": "Beijing"
}
response = requests.post("https://httpbin.org/post", data=post_data)

print(f"状态码：{response.status_code}")
data = response.json()
print(f"发送的数据：{data['form']}")

# 3. 发送 JSON 数据
print("\n3. 发送 JSON 数据")

json_data = {
    "username": "testuser",
    "password": "testpass"
}
response = requests.post("https://httpbin.org/post", json=json_data)

print(f"状态码：{response.status_code}")
data = response.json()
print(f"发送的 JSON: {data['json']}")

# 4. 请求头
print("\n4. 请求头")

headers = {
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json"
}
response = requests.get("https://httpbin.org/headers", headers=headers)

print(f"状态码：{response.status_code}")
data = response.json()
print(f"请求头中的 User-Agent: {data['headers']['User-Agent']}")

# 5. 处理响应
print("\n5. 处理响应")

response = requests.get("https://httpbin.org/html")
print(f"状态码：{response.status_code}")
print(f"响应内容长度：{len(response.text)}")
print(f"响应内容前 100 字符：{response.text[:100]}...")

# 6. 下载文件
print("\n6. 下载文件")

# 下载一个小图片作为示例
response = requests.get("https://httpbin.org/image/png")
if response.status_code == 200:
    with open("downloaded_image.png", "wb") as f:
        f.write(response.content)
    print("图片已下载到 downloaded_image.png")

# 7. 错误处理
print("\n7. 错误处理")

try:
    # 访问一个不存在的 URL
    response = requests.get("https://httpbin.org/status/404")
    response.raise_for_status()  # 如果状态码不是 200，会抛出异常
except requests.exceptions.HTTPError as e:
    print(f"HTTP 错误：{e}")

try:
    # 访问一个不存在的域名
    response = requests.get("https://nonexistent-domain-12345.com", timeout=5)
except requests.exceptions.RequestException as e:
    print(f"请求错误：{type(e).__name__}")

# 8. Session 对象
print("\n8. Session 对象")

session = requests.Session()
session.headers.update({"User-Agent": "MyApp/2.0"})

# 使用 session 发送多个请求，保持连接
response1 = session.get("https://httpbin.org/headers")
response2 = session.get("https://httpbin.org/headers")

print(f"第一个请求状态码：{response1.status_code}")
print(f"第二个请求状态码：{response2.status_code}")

session.close()

# 9. 常用状态码说明
print("\n=== 常用 HTTP 状态码 ===")
print("200 - 请求成功")
print("201 - 创建成功")
print("204 - 无内容")
print("301 - 永久重定向")
print("302 - 临时重定向")
print("400 - 请求错误")
print("401 - 未授权")
print("403 - 禁止访问")
print("404 - 未找到")
print("500 - 服务器内部错误")

print("\n提示：更多 requests 用法请参考官方文档 https://docs.python-requests.org/")
