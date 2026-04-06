# 03_functions.py
# Python 函数

# 1. 定义和调用函数
print("=== 定义和调用函数 ===")

def greet():
    """简单的问候函数"""
    print("Hello, Python!")

greet()

# 2. 带参数的函数
print("\n=== 带参数的函数 ===")

def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")
greet_person("Bob")

# 3. 带返回值的函数
print("\n=== 带返回值的函数 ===")

def add(a, b):
    return a + b

result = add(3, 5)
print(f"3 + 5 = {result}")

def multiply(x, y):
    return x * y

print(f"4 * 6 = {multiply(4, 6)}")

# 4. 多个返回值
print("\n=== 多个返回值 ===")

def get_info():
    return "Alice", 25, "Beijing"

name, age, city = get_info()
print(f"姓名: {name}, 年龄: {age}, 城市: {city}")

# 5. 默认参数
print("\n=== 默认参数 ===")

def greet_with_time(name, time="早上"):
    print(f"{time}好，{name}!")

greet_with_time("小明")
greet_with_time("小红", "晚上")

# 6. 关键字参数
print("\n=== 关键字参数 ===")

def describe_pet(pet_name, animal_type="dog"):
    print(f"我有一只{animal_type}，它叫{pet_name}")

describe_pet(pet_name="旺财")
describe_pet(animal_type="cat", pet_name="咪咪")

# 7. 可变参数
print("\n=== 可变参数 ===")

def sum_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

# 8. 可变关键字参数
print("\n=== 可变关键字参数 ===")

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Beijing")

# 9. lambda 匿名函数
print("\n=== lambda 匿名函数 ===")

square = lambda x: x ** 2
print(f"square(5) = {square(5)}")

add = lambda a, b: a + b
print(f"add(3, 7) = {add(3, 7)}")

# 10. 函数作为参数
print("\n=== 函数作为参数 ===")

def apply_operation(func, value):
    return func(value)

print(f"apply_operation(square, 4) = {apply_operation(square, 4)}")
