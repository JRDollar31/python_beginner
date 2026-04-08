# 11_matplotlib_demo.py
# Python Matplotlib 库基础操作

import matplotlib.pyplot as plt
import numpy as np

# Set English font (available on all Linux systems)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== Matplotlib Basics ===\n")

# 1. Simple line plot
print("1. Simple line plot...")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Sine Function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('line_plot.png', dpi=100)
print("Saved to line_plot.png")
plt.close()

# 2. Multiple lines
print("2. Multiple lines...")

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.title('Sine and Cosine')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('multi_line_plot.png', dpi=100)
print("Saved to multi_line_plot.png")
plt.close()

# 3. Scatter plot
print("3. Scatter plot...")

np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.rand(50) * 100
colors = np.random.rand(50)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis')
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Color value')
plt.savefig('scatter_plot.png', dpi=100)
print("Saved to scatter_plot.png")
plt.close()

# 4. Bar chart
print("4. Bar chart...")

categories = ['Apple', 'Banana', 'Orange', 'Grape', 'Watermelon']
values = [25, 40, 30, 15, 50]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['red', 'yellow', 'orange', 'purple', 'green'])
plt.title('Fruit Sales')
plt.xlabel('Fruit')
plt.ylabel('Sales')
plt.savefig('bar_plot.png', dpi=100)
print("Saved to bar_plot.png")
plt.close()

# 5. Horizontal bar chart
print("5. Horizontal bar chart...")

plt.figure(figsize=(10, 6))
plt.barh(categories, values, color=['red', 'yellow', 'orange', 'purple', 'green'])
plt.title('Fruit Sales (Horizontal)')
plt.xlabel('Sales')
plt.ylabel('Fruit')
plt.savefig('horizontal_bar_plot.png', dpi=100)
print("Saved to horizontal_bar_plot.png")
plt.close()

# 6. Pie chart
print("6. Pie chart...")

categories = ['Apple', 'Banana', 'Orange', 'Grape']
values = [30, 25, 25, 20]
colors_pie = ['red', 'yellow', 'orange', 'purple']
explode = [0.1, 0, 0, 0]

plt.figure(figsize=(8, 8))
plt.pie(values, labels=categories, colors=colors_pie, explode=explode, autopct='%1.1f%%', startangle=90)
plt.title('Fruit Sales Share')
plt.axis('equal')
plt.savefig('pie_plot.png', dpi=100)
print("Saved to pie_plot.png")
plt.close()

# 7. Histogram
print("7. Histogram...")

np.random.seed(42)
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Normal Distribution Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('histogram.png', dpi=100)
print("Saved to histogram.png")
plt.close()

# 8. Subplots
print("8. Subplots...")

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, np.cos(x), color='red')
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, np.sin(x) * np.cos(x), color='green')
axes[1, 0].set_title('sin(x) * cos(x)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, x ** 2, color='purple')
axes[1, 1].set_title('x^2')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subplots.png', dpi=100)
print("Saved to subplots.png")
plt.close()

# 9. Styled plot
print("9. Styled plot...")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2, linestyle='-', marker='o', markersize=3)
plt.title('Styled Line Plot', fontsize=16)
plt.xlabel('X axis', fontsize=14)
plt.ylabel('Y axis', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig('styled_plot.png', dpi=100)
print("Saved to styled_plot.png")
plt.close()

print("\nAll plots saved to current directory!")
print("For more: https://matplotlib.org/")
