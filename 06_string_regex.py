# 06_string_regex.py
# Python 字符串和正则表达式

import re

# ========== 字符串操作 ==========
print("=== 字符串基本操作 ===")

# 1. 字符串创建
text = "Hello, Python World!"
print(f"原字符串: {text}")

# 2. 字符串索引和切片
print("\n=== 字符串索引和切片 ===")
print(f"text[0]: {text[0]}")
print(f"text[7:14]: {text[7:14]}")
print(f"text[-1]: {text[-1]}")
print(f"text[::-1]: {text[::-1]}")

# 3. 字符串连接和格式化
print("\n=== 字符串连接和格式化 ===")
name = "Alice"
age = 25
print("连接: " + "Hello, " + name)
print(f"f-string: {name} is {age} years old")
print("format: {} is {} years old".format(name, age))

# 4. 常用字符串方法
print("\n=== 常用字符串方法 ===")
text = "  Hello, World!  "
print(f"原字符串: '{text}'")
print(f"strip(): '{text.strip()}'")
print(f"upper(): '{text.upper()}'")
print(f"lower(): '{text.lower()}'")
print(f"replace('World', 'Python'): '{text.replace('World', 'Python')}'")

# 5. 字符串分割和连接
print("\n=== 字符串分割和连接 ===")
sentence = "apple,banana,orange"
fruits = sentence.split(",")
print(f"split 后: {fruits}")

joined = "-".join(fruits)
print(f"join 后: {joined}")

# 6. 字符串查找
print("\n=== 字符串查找 ===")
text = "Hello, Python!"
print(f"find('Python'): {text.find('Python')}")
print(f"find('Java'): {text.find('Java')}")
print(f"'Python' in text: {'Python' in text}")
print(f"text.startswith('Hello'): {text.startswith('Hello')}")
print(f"text.endswith('!'): {text.endswith('!')}")

# 7. 其他字符串方法
print("\n=== 其他字符串方法 ===")
text = "python123"
print(f"text.isdigit(): {'123'.isdigit()}")
print(f"text.isalpha(): {'abc'.isalpha()}")
print(f"text.isalnum(): {text.isalnum()}")
print(f"'hello'.capitalize(): {'hello'.capitalize()}")
print(f"'hello world'.title(): {'hello world'.title()}")


# ========== 正则表达式 ==========
print("\n\n=== 正则表达式 ===")

# 1. 基本匹配
print("\n基本匹配:")
pattern = r"hello"
text = "hello world"
match = re.search(pattern, text)
print(f"搜索 '{pattern}' 在 '{text}': {match.group() if match else '未找到'}")

# 2. 匹配数字
print("\n匹配数字:")
pattern = r"\d+"
text = "电话号码：123-4567"
matches = re.findall(pattern, text)
print(f"找到的数字: {matches}")

# 3. 匹配邮箱
print("\n匹配邮箱:")
pattern = r"\w+@\w+\.\w+"
text = "请联系 admin@example.com 或 support@test.org"
emails = re.findall(pattern, text)
print(f"找到的邮箱: {emails}")

# 4. 匹配电话号码
print("\n匹配电话号码:")
pattern = r"\d{3}-\d{4}"
text = "电话：123-4567 和 890-1234"
phones = re.findall(pattern, text)
print(f"找到的电话: {phones}")

# 5. 常用正则表达式符号
print("\n=== 常用正则表达式符号 ===")
print(r"\d - 匹配数字 (0-9)")
print(r"\w - 匹配字母、数字或下划线")
print(r"\s - 匹配空白字符")
print(r".  - 匹配任意字符")
print(r"*  - 匹配 0 次或多次")
print(r"+  - 匹配 1 次或多次")
print(r"?  - 匹配 0 次或 1 次")
print(r"^  - 匹配字符串开头")
print(r"$  - 匹配字符串结尾")

# 6. 替换
print("\n=== 正则替换 ===")
text = "2024-01-15"
pattern = r"-"
replacement = "/"
result = re.sub(pattern, replacement, text)
print(f"替换后: {result}")

# 7. 分组
print("\n=== 正则分组 ===")
text = "2024-01-15"
pattern = r"(\d{4})-(\d{2})-(\d{2})"
match = re.match(pattern, text)
if match:
    print(f"年：{match.group(1)}")
    print(f"月：{match.group(2)}")
    print(f"日：{match.group(3)}")
