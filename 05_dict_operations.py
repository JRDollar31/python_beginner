# 05_dict_operations.py
# Python 字典操作

# 1. 创建字典
print("=== 创建字典 ===")
person = {
    "name": "Alice",
    "age": 25,
    "city": "Beijing"
}
empty_dict = {}
another_dict = dict(name="Bob", age=30)

print(f"person 字典: {person}")
print(f"空字典: {empty_dict}")
print(f"another_dict: {another_dict}")

# 2. 访问字典元素
print("\n=== 访问字典元素 ===")
print(f"姓名: {person['name']}")
print(f"年龄: {person['age']}")
print(f"使用 get 访问: {person.get('city')}")
print(f"访问不存在的键 (返回 None): {person.get('country')}")
print(f"访问不存在的键 (返回默认值): {person.get('country', 'China')}")

# 3. 修改字典元素
print("\n=== 修改字典元素 ===")
person["age"] = 26
print(f"修改年龄后: {person}")

person["job"] = "Engineer"
print(f"添加新键值对后: {person}")

# 4. 删除字典元素
print("\n=== 删除字典元素 ===")
person = {"name": "Alice", "age": 25, "city": "Beijing", "job": "Engineer"}

removed = person.pop("job")      # 删除并返回值
print(f"pop 后: {person}, 删除的值: {removed}")

del person["city"]               # 删除键值对
print(f"del 后: {person}")

person.clear()                   # 清空字典
print(f"clear 后: {person}")

# 5. 字典查询
print("\n=== 字典查询 ===")
person = {"name": "Alice", "age": 25, "city": "Beijing"}
print(f"'name' 在字典中: {'name' in person}")
print(f"'country' 在字典中: {'country' in person}")
print(f"字典长度: {len(person)}")

# 6. 遍历字典
print("\n=== 遍历字典 ===")
person = {"name": "Alice", "age": 25, "city": "Beijing"}

print("遍历键:")
for key in person.keys():
    print(f"  键: {key}")

print("\n遍历值:")
for value in person.values():
    print(f"  值: {value}")

print("\n遍历键值对:")
for key, value in person.items():
    print(f"  {key}: {value}")

# 7. 字典常用方法
print("\n=== 字典常用方法 ===")
person = {"name": "Alice", "age": 25}

# get - 获取值
print(f"get('name'): {person.get('name')}")

# setdefault - 设置默认值
person.setdefault("city", "Beijing")
print(f"setdefault 后: {person}")

# update - 更新字典
person.update({"age": 26, "job": "Engineer"})
print(f"update 后: {person}")

# copy - 复制字典
person_copy = person.copy()
print(f"复制的字典: {person_copy}")

# 8. 字典推导式
print("\n=== 字典推导式 ===")
squares = {x: x ** 2 for x in range(1, 6)}
print(f"平方字典: {squares}")

# 创建键值互换的字典
original = {"a": 1, "b": 2, "c": 3}
swapped = {value: key for key, value in original.items()}
print(f"键值互换: {swapped}")

# 9. 嵌套字典
print("\n=== 嵌套字典 ===")
students = {
    "student1": {"name": "Alice", "age": 20},
    "student2": {"name": "Bob", "age": 21},
    "student3": {"name": "Charlie", "age": 22}
}

print(f"学生 1 的姓名: {students['student1']['name']}")
print(f"学生 2 的年龄: {students['student2']['age']}")

# 遍历嵌套字典
for student_id, info in students.items():
    print(f"  {student_id}: {info['name']}, {info['age']}岁")
