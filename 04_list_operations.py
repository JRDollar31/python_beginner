# 04_list_operations.py
# Python 列表（数组）操作

# 1. 创建列表
print("=== 创建列表 ===")
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
empty_list = []

print(f"水果列表: {fruits}")
print(f"数字列表: {numbers}")
print(f"空列表: {empty_list}")

# 2. 访问列表元素
print("\n=== 访问列表元素 ===")
print(f"第一个水果: {fruits[0]}")
print(f"第二个水果: {fruits[1]}")
print(f"最后一个水果: {fruits[-1]}")
print(f"倒数第二个水果: {fruits[-2]}")

# 3. 修改列表元素
print("\n=== 修改列表元素 ===")
fruits[1] = "grape"
print(f"修改后: {fruits}")

# 4. 列表切片
print("\n=== 列表切片 ===")
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"原列表: {numbers}")
print(f"numbers[2:5]: {numbers[2:5]}")
print(f"numbers[:4]: {numbers[:4]}")
print(f"numbers[6:]: {numbers[6:]}")
print(f"numbers[::2]: {numbers[::2]}")
print(f"numbers[::-1]: {numbers[::-1]}")

# 5. 添加元素
print("\n=== 添加元素 ===")
fruits = ["apple", "banana"]
fruits.append("orange")          # 添加到末尾
print(f"append 后: {fruits}")

fruits.insert(1, "grape")        # 插入到指定位置
print(f"insert 后: {fruits}")

fruits.extend(["melon", "pear"]) # 扩展列表
print(f"extend 后: {fruits}")

# 6. 删除元素
print("\n=== 删除元素 ===")
fruits = ["apple", "banana", "orange", "grape"]
fruits.remove("banana")          # 删除指定值
print(f"remove 后: {fruits}")

popped = fruits.pop()            # 删除并返回最后一个元素
print(f"pop 后: {fruits}, 弹出的元素: {popped}")

popped = fruits.pop(1)           # 删除并返回指定位置元素
print(f"pop(1) 后: {fruits}, 弹出的元素: {popped}")

# 7. 列表查询
print("\n=== 列表查询 ===")
fruits = ["apple", "banana", "orange", "apple"]
print(f"'banana' 在列表中: {'banana' in fruits}")
print(f"'melon' 在列表中: {'melon' in fruits}")
print(f"'apple' 出现次数: {fruits.count('apple')}")
print(f"'orange' 的索引: {fruits.index('orange')}")

# 8. 列表排序
print("\n=== 列表排序 ===")
numbers = [3, 1, 4, 1, 5, 9, 2]
print(f"原列表: {numbers}")

numbers.sort()                   # 升序排序
print(f"sort 后: {numbers}")

numbers.sort(reverse=True)       # 降序排序
print(f"降序排序后: {numbers}")

# 9. 其他列表操作
print("\n=== 其他列表操作 ===")
numbers = [1, 2, 3]
print(f"列表长度: {len(numbers)}")
print(f"最大值: {max(numbers)}")
print(f"最小值: {min(numbers)}")
print(f"求和: {sum(numbers)}")

# 10. 列表推导式
print("\n=== 列表推导式 ===")
squares = [x ** 2 for x in range(1, 6)]
print(f"平方列表: {squares}")

evens = [x for x in range(10) if x % 2 == 0]
print(f"偶数列表: {evens}")

# 11. 列表拼接和重复
print("\n=== 列表拼接和重复 ===")
list1 = [1, 2, 3]
list2 = [4, 5, 6]
print(f"列表拼接: {list1 + list2}")
print(f"列表重复: {list1 * 3}")
