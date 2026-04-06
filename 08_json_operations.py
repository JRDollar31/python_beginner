# 08_json_operations.py
# Python JSON 操作

import json

# 1. JSON 字符串解析（反序列化）
print("=== JSON 字符串解析 ===")

json_string = '{"name": "Alice", "age": 25, "city": "Beijing"}'
data = json.loads(json_string)
print(f"解析后的数据: {data}")
print(f"姓名: {data['name']}")
print(f"年龄: {data['age']}")

# 2. Python 对象转 JSON 字符串（序列化）
print("\n=== Python 对象转 JSON 字符串 ===")

person = {
    "name": "Bob",
    "age": 30,
    "city": "Shanghai"
}
json_string = json.dumps(person)
print(f"JSON 字符串: {json_string}")
print(f"类型: {type(json_string)}")

# 3. 格式化 JSON 输出
print("\n=== 格式化 JSON 输出 ===")

person = {
    "name": "Charlie",
    "age": 28,
    "hobbies": ["reading", "swimming", "coding"],
    "address": {
        "city": "Guangzhou",
        "district": "Tianhe"
    }
}

# 使用 indent 参数格式化
json_string = json.dumps(person, indent=2, ensure_ascii=False)
print("格式化后的 JSON:")
print(json_string)

# 4. 列表转 JSON
print("\n=== 列表转 JSON ===")

fruits = ["apple", "banana", "orange"]
json_string = json.dumps(fruits, ensure_ascii=False)
print(f"列表转 JSON: {json_string}")

numbers = [1, 2, 3, 4, 5]
json_string = json.dumps(numbers)
print(f"数字列表转 JSON: {json_string}")

# 5. 读写 JSON 文件
print("\n=== 读写 JSON 文件 ===")

# 写入 JSON 文件
data = {
    "students": [
        {"name": "Alice", "age": 20, "grade": "A"},
        {"name": "Bob", "age": 21, "grade": "B"},
        {"name": "Charlie", "age": 22, "grade": "A"}
    ],
    "school": "第一中学"
}

with open("students.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("数据已写入 students.json 文件")

# 读取 JSON 文件
with open("students.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)
print(f"\n从文件读取的数据:")
print(f"学校: {loaded_data['school']}")
print(f"学生数量: {len(loaded_data['students'])}")
for student in loaded_data['students']:
    print(f"  - {student['name']}, {student['age']}岁，成绩：{student['grade']}")

# 6. Python 类型与 JSON 类型对应
print("\n=== Python 类型与 JSON 类型对应 ===")
print("Python  dict    ->  JSON  object")
print("Python  list    ->  JSON  array")
print("Python  str     ->  JSON  string")
print("Python  int     ->  JSON  number")
print("Python  float   ->  JSON  number")
print("Python  bool    ->  JSON  boolean")
print("Python  None    ->  JSON  null")

# 7. 处理复杂嵌套结构
print("\n=== 处理复杂嵌套结构 ===")

complex_data = {
    "company": "科技公司",
    "departments": [
        {
            "name": "技术部",
            "employees": [
                {"name": "张三", "position": "工程师"},
                {"name": "李四", "position": "架构师"}
            ]
        },
        {
            "name": "市场部",
            "employees": [
                {"name": "王五", "position": "经理"}
            ]
        }
    ]
}

json_string = json.dumps(complex_data, indent=2, ensure_ascii=False)
print("复杂嵌套结构:")
print(json_string)

# 解析后访问
data = json.loads(json_string)
print(f"\n技术部员工:")
for dept in data['departments']:
    if dept['name'] == "技术部":
        for emp in dept['employees']:
            print(f"  - {emp['name']}: {emp['position']}")
