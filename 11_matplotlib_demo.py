# 11_matplotlib_demo.py
# Python Matplotlib 库基础操作

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（根据系统选择合适的字体）
# Windows 系统可以使用 'SimHei'
# Mac 系统可以使用 'Arial Unicode MS'
# Linux 系统可能需要安装中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("=== Matplotlib 基础操作 ===\n")

# 1. 简单的折线图
print("1. 绘制折线图...")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('正弦函数图像')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('line_plot.png', dpi=100)
print("折线图已保存到 line_plot.png")
plt.close()

# 2. 多条折线
print("2. 绘制多条折线...")

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.title('正弦和余弦函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('multi_line_plot.png', dpi=100)
print("多条折线图已保存到 multi_line_plot.png")
plt.close()

# 3. 散点图
print("3. 绘制散点图...")

np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 100
colors = np.random.rand(50)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis')
plt.title('散点图')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='颜色值')
plt.savefig('scatter_plot.png', dpi=100)
print("散点图已保存到 scatter_plot.png")
plt.close()

# 4. 柱状图
print("4. 绘制柱状图...")

categories = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
values = [25, 40, 30, 15, 50]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['red', 'yellow', 'orange', 'purple', 'green'])
plt.title('水果销量统计')
plt.xlabel('水果')
plt.ylabel('销量')
plt.savefig('bar_plot.png', dpi=100)
print("柱状图已保存到 bar_plot.png")
plt.close()

# 5. 水平柱状图
print("5. 绘制水平柱状图...")

plt.figure(figsize=(10, 6))
plt.barh(categories, values, color=['red', 'yellow', 'orange', 'purple', 'green'])
plt.title('水果销量统计（水平）')
plt.xlabel('销量')
plt.ylabel('水果')
plt.savefig('horizontal_bar_plot.png', dpi=100)
print("水平柱状图已保存到 horizontal_bar_plot.png")
plt.close()

# 6. 饼图
print("6. 绘制饼图...")

categories = ['苹果', '香蕉', '橙子', '葡萄']
values = [30, 25, 25, 20]
colors = ['red', 'yellow', 'orange', 'purple']
explode = [0.1, 0, 0, 0]  # 突出显示第一个扇形

plt.figure(figsize=(8, 8))
plt.pie(values, labels=categories, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
plt.title('水果销量占比')
plt.axis('equal')  # 保证饼图是正圆
plt.savefig('pie_plot.png', dpi=100)
print("饼图已保存到 pie_plot.png")
plt.close()

# 7. 直方图
print("7. 绘制直方图...")

np.random.seed(42)
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('正态分布直方图')
plt.xlabel('值')
plt.ylabel('频数')
plt.savefig('histogram.png', dpi=100)
print("直方图已保存到 histogram.png")
plt.close()

# 8. 子图
print("8. 绘制子图...")

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 第一个子图
axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# 第二个子图
axes[0, 1].plot(x, np.cos(x), color='red')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

# 第三个子图
axes[1, 0].plot(x, np.sin(x) * np.cos(x), color='green')
axes[1, 0].set_title('sin(x) * cos(x)')
axes[1, 0].grid(True, alpha=0.3)

# 第四个子图
axes[1, 1].plot(x, x ** 2, color='purple')
axes[1, 1].set_title('x^2')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subplots.png', dpi=100)
print("子图已保存到 subplots.png")
plt.close()

# 9. 设置图形样式
print("9. 设置图形样式...")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2, linestyle='-', marker='o', markersize=3)
plt.title('带样式的折线图', fontsize=16)
plt.xlabel('x 轴', fontsize=14)
plt.ylabel('y 轴', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig('styled_plot.png', dpi=100)
print("样式图已保存到 styled_plot.png")
plt.close()

print("\n所有图形已保存到当前目录！")
print("提示：更多 Matplotlib 用法请参考官方文档 https://matplotlib.org/")
