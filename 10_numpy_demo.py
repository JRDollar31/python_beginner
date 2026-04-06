# 10_numpy_demo.py
# Python NumPy 库基础操作

import numpy as np

print("=== NumPy 基础操作 ===\n")

# 1. 创建数组
print("1. 创建数组")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组：{arr1}")

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2}")

# 使用 arange 创建
arr3 = np.arange(0, 10, 2)
print(f"arange(0, 10, 2): {arr3}")

# 使用 zeros 创建
arr4 = np.zeros((3, 3))
print(f"zeros((3, 3)):\n{arr4}")

# 使用 ones 创建
arr5 = np.ones((2, 4))
print(f"ones((2, 4)):\n{arr5}")

# 使用 full 创建
arr6 = np.full((2, 2), 7)
print(f"full((2, 2), 7):\n{arr6}")

# 使用 eye 创建单位矩阵
arr7 = np.eye(3)
print(f"eye(3):\n{arr7}")

# 使用 random 创建随机数组
arr8 = np.random.rand(2, 3)
print(f"random.rand(2, 3):\n{arr8}")

# 2. 数组属性
print("\n2. 数组属性")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"数组:\n{arr}")
print(f"形状 (shape): {arr.shape}")
print(f"维度 (ndim): {arr.ndim}")
print(f"元素总数 (size): {arr.size}")
print(f"数据类型 (dtype): {arr.dtype}")

# 3. 数组索引和切片
print("\n3. 数组索引和切片")

arr = np.array([10, 20, 30, 40, 50])
print(f"数组：{arr}")
print(f"arr[0]: {arr[0]}")
print(f"arr[-1]: {arr[-1]}")
print(f"arr[1:4]: {arr[1:4]}")
print(f"arr[::2]: {arr[::2]}")

# 二维数组索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组:\n{arr2d}")
print(f"arr2d[0, 1]: {arr2d[0, 1]}")
print(f"arr2d[1, :]: {arr2d[1, :]}")
print(f"arr2d[:, 2]: {arr2d[:, 2]}")

# 4. 数组运算
print("\n4. 数组运算")

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 - arr2: {arr1 - arr2}")
print(f"arr1 * arr2: {arr1 * arr2}")
print(f"arr1 / arr2: {arr1 / arr2}")
print(f"arr1 ** 2: {arr1 ** 2}")

# 5. 数组函数
print("\n5. 数组函数")

arr = np.array([1, 2, 3, 4, 5])
print(f"数组：{arr}")
print(f"sum(): {np.sum(arr)}")
print(f"mean(): {np.mean(arr)}")
print(f"max(): {np.max(arr)}")
print(f"min(): {np.min(arr)}")
print(f"std(): {np.std(arr)}")  # 标准差
print(f"argmax(): {np.argmax(arr)}")  # 最大值索引

# 6. 数组形状操作
print("\n6. 数组形状操作")

arr = np.array([1, 2, 3, 4, 5, 6])
print(f"原数组：{arr}")

# reshape 改变形状
arr_reshaped = arr.reshape(2, 3)
print(f"reshape(2, 3):\n{arr_reshaped}")

arr_reshaped = arr.reshape(3, 2)
print(f"reshape(3, 2):\n{arr_reshaped}")

# flatten 展平
arr_flat = arr_reshaped.flatten()
print(f"flatten(): {arr_flat}")

# transpose 转置
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n原数组:\n{arr2d}")
print(f"转置:\n{arr2d.T}")

# 7. 数组拼接
print("\n7. 数组拼接")

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"arr1:\n{arr1}")
print(f"arr2:\n{arr2}")

print(f"\n垂直拼接 (vstack):\n{np.vstack((arr1, arr2))}")
print(f"\n水平拼接 (hstack):\n{np.hstack((arr1, arr2))}")

# 8. 布尔索引
print("\n8. 布尔索引")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"原数组：{arr}")

# 找出大于 5 的元素
print(f"大于 5 的元素：{arr[arr > 5]}")

# 找出偶数
print(f"偶数：{arr[arr % 2 == 0]}")

# 9. 线性代数操作（简单了解）
print("\n9. 线性代数操作")

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"arr1:\n{arr1}")
print(f"arr2:\n{arr2}")

# 矩阵乘法
print(f"\n矩阵乘法 (dot):\n{np.dot(arr1, arr2)}")

# 10. 广播机制
print("\n10. 广播机制")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原数组:\n{arr}")
print(f"\n加 10:\n{arr + 10}")
print(f"\n乘 2:\n{arr * 2}")
