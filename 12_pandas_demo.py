# 12_pandas_demo.py
# Python Pandas 数据分析库基础操作

import pandas as pd
import numpy as np

print("=== Pandas 基础操作 ===\n")

# 1. Series 数据结构
print("1. Series 数据结构")

# 从列表创建 Series
s = pd.Series([10, 20, 30, 40, 50])
print(f"Series:\n{s}")

# 带索引的 Series
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(f"\n带索引的 Series:\n{s2}")
print(f"s2['a']: {s2['a']}")

# 从字典创建 Series
data = {"name": "Alice", "age": 25, "city": "Beijing"}
s3 = pd.Series(data)
print(f"\n从字典创建的 Series:\n{s3}")

# 2. DataFrame 数据结构
print("\n2. DataFrame 数据结构")

# 从字典创建 DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 35, 28],
    "city": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
    "score": [85, 92, 78, 90]
})
print(f"DataFrame:\n{df}")

# 从列表的字典创建
data2 = [
    {"name": "Eve", "age": 22, "score": 88},
    {"name": "Frank", "age": 33, "score": 76}
]
df2 = pd.DataFrame(data2)
print(f"\n从列表字典创建:\n{df2}")

# 3. 查看数据
print("\n3. 查看数据")

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 28, 22],
    "city": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"],
    "score": [85, 92, 78, 90, 95]
})

print(f"前 3 行:\n{df.head(3)}")
print(f"\n后 2 行:\n{df.tail(2)}")
print(f"\n基本信息:\n{df.info()}")
print(f"\n统计摘要:\n{df.describe()}")

# 4. 选择列和行
print("\n4. 选择列和行")

# 选择单列
print(f"选择 'name' 列:\n{df['name']}")

# 选择多列
print(f"\n选择多列:\n{df[['name', 'score']]}")

# 使用 loc 按标签选择
print(f"\nloc 选择第 0 行:\n{df.loc[0]}")
print(f"\nloc 选择多行:\n{df.loc[0:2, ['name', 'age']]}")

# 使用 iloc 按位置选择
print(f"\niloc 选择第 1 行:\n{df.iloc[1]}")
print(f"\niloc 切片:\n{df.iloc[1:4, 0:2]}")

# 5. 条件过滤
print("\n5. 条件过滤")

print(f"分数大于 85:\n{df[df['score'] > 85]}")
print(f"\n年龄小于 30:\n{df[df['age'] < 30]}")
print(f"\n多条件 (分数>80 且 年龄<30):\n{df[(df['score'] > 80) & (df['age'] < 30)]}")

# 6. 添加和修改列
print("\n6. 添加和修改列")

df_copy = df.copy()

# 添加新列
df_copy["grade"] = df_copy["score"].apply(lambda x: "A" if x >= 90 else "B" if x >= 80 else "C")
print(f"添加 grade 列:\n{df_copy}")

# 修改列
df_copy["score"] = df_copy["score"] + 5
print(f"\n分数 +5 后:\n{df_copy}")

# 删除列
df_copy2 = df_copy.drop(columns=["city"])
print(f"\n删除 city 列:\n{df_copy2}")

# 7. 排序
print("\n7. 排序")

print(f"按分数降序:\n{df.sort_values('score', ascending=False)}")
print(f"\n按年龄升序:\n{df.sort_values('age', ascending=True)}")

# 8. 分组聚合
print("\n8. 分组聚合")

df_group = pd.DataFrame({
    "department": ["Sales", "Tech", "Sales", "Tech", "HR", "HR"],
    "employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "salary": [5000, 8000, 5500, 9000, 4500, 4800]
})

print(f"按部门分组求平均:\n{df_group.groupby('department')['salary'].mean()}")
print(f"\n按部门分组求和:\n{df_group.groupby('department')['salary'].sum()}")
print(f"\n分组统计:\n{df_group.groupby('department')['salary'].agg(['mean', 'sum', 'count'])}")

# 9. 缺失值处理
print("\n9. 缺失值处理")

df_na = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": [1, 2, 3, 4, 5]
})
print(f"含缺失值的 DataFrame:\n{df_na}")
print(f"\n检测缺失值:\n{df_na.isnull()}")
print(f"\n每列缺失值数量:\n{df_na.isnull().sum()}")
print(f"\n删除含缺失值的行:\n{df_na.dropna()}")
print(f"\n填充缺失值为 0:\n{df_na.fillna(0)}")
print(f"\n用均值填充:\n{df_na.fillna(df_na.mean())}")

# 10. 合并 DataFrame
print("\n10. 合并 DataFrame")

df1 = pd.DataFrame({
    "key": ["A", "B", "C"],
    "value1": [1, 2, 3]
})
df2 = pd.DataFrame({
    "key": ["A", "B", "D"],
    "value2": [4, 5, 6]
})

print(f"df1:\n{df1}")
print(f"\ndf2:\n{df2}")
print(f"\n内连接 merge:\n{pd.merge(df1, df2, on='key', how='inner')}")
print(f"\n外连接 merge:\n{pd.merge(df1, df2, on='key', how='outer')}")
print(f"\n左连接 merge:\n{pd.merge(df1, df2, on='key', how='left')}")

# 使用 concat 拼接
df3 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df4 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
print(f"\nconcat 垂直拼接:\n{pd.concat([df3, df4], ignore_index=True)}")

# 11. 读写文件
print("\n11. 读写文件")

# 写入 CSV
df.to_csv("temp_output.csv", index=False)
print("已写入 temp_output.csv")

# 读取 CSV
df_read = pd.read_csv("temp_output.csv")
print(f"读取的 DataFrame:\n{df_read}")

# 写入/读取 Excel (需要 openpyxl)
try:
    df.to_excel("temp_output.xlsx", index=False)
    df_excel = pd.read_excel("temp_output.xlsx")
    print(f"读取的 Excel DataFrame:\n{df_excel}")
except ImportError:
    print("openpyxl 未安装，跳过 Excel 操作 (pip install openpyxl)")

# 12. 字符串操作
print("\n12. 字符串操作")

df_str = pd.DataFrame({
    "text": ["Hello World", "Python Pandas", "Data Analysis", "Machine Learning"]
})
print(f"原数据:\n{df_str}")
print(f"\n转小写:\n{df_str['text'].str.lower()}")
print(f"\n按空格分割:\n{df_str['text'].str.split(' ')}")
print(f"\n包含 'Data':\n{df_str['text'].str.contains('Data')}")
print(f"\n字符长度:\n{df_str['text'].str.len()}")

# 13. 时间序列
print("\n13. 时间序列")

df_time = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=5, freq="D"),
    "value": [100, 200, 150, 300, 250]
})
print(f"时间序列数据:\n{df_time}")
print(f"\n提取年份:\n{df_time['date'].dt.year}")
print(f"\n提取月份:\n{df_time['date'].dt.month}")
print(f"\n提取星期:\n{df_time['date'].dt.day_name()}")

# 清理临时文件
import os
if os.path.exists("temp_output.csv"):
    os.remove("temp_output.csv")
if os.path.exists("temp_output.xlsx"):
    os.remove("temp_output.xlsx")

print("\n=== Pandas 基础操作完成 ===")
