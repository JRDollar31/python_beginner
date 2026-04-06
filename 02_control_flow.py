# 02_control_flow.py
# Python 条件语句和循环

# 1. if-else 条件语句
print("=== if-else 条件语句 ===")
score = 85

if score >= 90:
    print("优秀！")
elif score >= 80:
    print("良好！")
elif score >= 60:
    print("及格！")
else:
    print("不及格！")

# 2. 比较运算符
print("\n=== 比较运算符 ===")
x = 10
y = 20
print(f"{x} == {y}: {x == y}")
print(f"{x} != {y}: {x != y}")
print(f"{x} < {y}: {x < y}")
print(f"{x} > {y}: {x > y}")
print(f"{x} <= {y}: {x <= y}")
print(f"{x} >= {y}: {x >= y}")

# 3. for 循环
print("\n=== for 循环 ===")

# 遍历列表
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"水果: {fruit}")

# 使用 range
print("\n使用 range(5):")
for i in range(5):
    print(i, end=" ")

print("\n\n使用 range(2, 6):")
for i in range(2, 6):
    print(i, end=" ")

print("\n\n使用 range(1, 10, 2):")
for i in range(1, 10, 2):
    print(i, end=" ")

# 4. while 循环
print("\n\n=== while 循环 ===")
count = 0
while count < 5:
    print(f"count = {count}")
    count += 1

# 5. break 和 continue
print("\n=== break 和 continue ===")

# break - 跳出循环
print("break 示例:")
for i in range(10):
    if i == 5:
        break
    print(i, end=" ")

# continue - 跳过本次循环
print("\n\ncontinue 示例:")
for i in range(5):
    if i == 2:
        continue
    print(i, end=" ")

# 6. 嵌套循环
print("\n\n=== 嵌套循环 ===")
for i in range(3):
    for j in range(3):
        print(f"({i}, {j})", end=" ")
    print()

# 7. 列表推导式（简单了解）
print("\n=== 列表推导式 ===")
squares = [x ** 2 for x in range(5)]
print(f"平方列表: {squares}")

evens = [x for x in range(10) if x % 2 == 0]
print(f"偶数列表: {evens}")
