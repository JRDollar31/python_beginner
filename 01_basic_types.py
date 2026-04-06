# 01_basic_types.py
# Python 基础数据类型

# 1. 数字类型
integer_num = 42          # 整数
float_num = 3.14          # 浮点数
negative_num = -10        # 负数

print( "整数: " + str(integer_num))
print(f"整数: {integer_num}")

print("=== 数字类型 ===")
print(f"整数: {integer_num}")
print(f"浮点数: {float_num}")
print(f"负数: {negative_num}")

# 2. 基本运算
a = 10
b = 3

print("\n=== 基本运算 ===")
print(f"{a} + {b} = {a + b}")      # 加法
print(f"{a} - {b} = {a - b}")      # 减法
print(f"{a} * {b} = {a * b}")      # 乘法
print(f"{a} / {b} = {a / b}")      # 除法
print(f"{a} // {b} = {a // b}")    # 整除
print(f"{a} % {b} = {a % b}")      # 取余
print(f"{a} ** {b} = {a ** b}")    # 幂运算

# 3. 字符串
name = "Python"
greeting = 'Hello, World!'
multi_line = """这是
多行
字符串"""

print("\n=== 字符串 ===")
print(f"字符串: {name}")
print(f"连接: {'Hi ' + name}")
print(f"重复: {name * 2}")
print(f"长度: {len(name)}")
print(f"大写: {name.upper()}")
print(f"小写: {name.lower()}")

# 4. 布尔类型
is_true = True
is_false = False

print("\n=== 布尔类型 ===")
print(f"True and False: {is_true and is_false}")
print(f"True or False: {is_true or is_false}")
print(f"not True: {not is_true}")

# 5. 列表（数组）
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

print("\n=== 列表 ===")
print(f"水果列表: {fruits}")
print(f"第一个水果: {fruits[0]}")
print(f"最后一个水果: {fruits[-1]}")
print(f"列表长度: {len(fruits)}")

# 6. 字典
person = {
    "name": "Alice",
    "age": 25,
    "city": "Beijing"
}

print("\n=== 字典 ===")
print(f"个人信息: {person}")
print(f"姓名: {person['name']}")
print(f"年龄: {person['age']}")

# 7. None 类型
nothing = None
print("\n=== None 类型 ===")
print(f"None 值: {nothing}")
