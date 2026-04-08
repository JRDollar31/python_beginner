# 13_scipy_demo.py
# Python SciPy 科学计算库基础操作

from scipy import stats, optimize, integrate, interpolate, linalg, spatial
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== SciPy 基础操作 ===\n")

# 1. 统计模块 (scipy.stats)
print("1. 统计模块 - 概率分布")

# 正态分布
norm_dist = stats.norm(loc=0, scale=1)  # 均值=0, 标准差=1
print(f"正态分布 N(0,1):")
print(f"  PDF at 0: {norm_dist.pdf(0):.4f}")
print(f"  CDF at 0: {norm_dist.cdf(0):.4f}")
print(f"  中位数: {norm_dist.ppf(0.5):.4f}")
print(f"  95% 分位数: {norm_dist.ppf(0.95):.4f}")

# 生成随机样本
samples = norm_dist.rvs(1000)
print(f"\n随机样本统计:")
print(f"  均值: {np.mean(samples):.4f}")
print(f"  标准差: {np.std(samples):.4f}")

# 其他常见分布
print(f"\n均匀分布 U(0,1): {stats.uniform.rvs(size=5)}")
print(f"t 分布 (df=10): {stats.t.rvs(df=10, size=5)}")
print(f"卡方分布 (df=5): {stats.chi2.rvs(df=5, size=5)}")

# 2. 描述性统计
print("\n2. 描述性统计")

data = [10, 20, 20, 30, 30, 30, 40, 40, 50]
print(f"数据: {data}")
print(f"均值: {np.mean(data)}")
print(f"中位数: {np.median(data)}")
print(f"标准差: {np.std(data):.2f}")
print(f"偏度: {stats.skew(data):.4f}")
print(f"峰度: {stats.kurtosis(data):.4f}")

# 3. 假设检验
print("\n3. 假设检验")

# 单样本 t 检验
sample_data = stats.norm.rvs(loc=50, scale=10, size=100, random_state=42)
t_stat, p_value = stats.ttest_1samp(sample_data, popmean=50)
print(f"单样本 t 检验 (H0: μ=50):")
print(f"  t 统计量: {t_stat:.4f}")
print(f"  p 值: {p_value:.4f}")
print(f"  结论: {'不拒绝 H0' if p_value > 0.05 else '拒绝 H0'}")

# 两独立样本 t 检验
group1 = stats.norm.rvs(loc=10, scale=2, size=50, random_state=42)
group2 = stats.norm.rvs(loc=12, scale=2, size=50, random_state=43)
t_stat2, p_value2 = stats.ttest_ind(group1, group2)
print(f"\n两独立样本 t 检验:")
print(f"  t 统计量: {t_stat2:.4f}")
print(f"  p 值: {p_value2:.4f}")
print(f"  结论: {'两组有显著差异' if p_value2 < 0.05 else '两组无显著差异'}")

# 4. 线性回归
print("\n4. 线性回归")

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 8.1, 9.8])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"线性回归结果:")
print(f"  斜率: {slope:.4f}")
print(f"  截距: {intercept:.4f}")
print(f"  R²: {r_value**2:.4f}")
print(f"  p 值: {p_value:.4f}")
print(f"  回归方程: y = {slope:.2f}x + {intercept:.2f}")

# 5. 优化模块 (scipy.optimize)
print("\n5. 优化模块 - 求函数最小值")

# 定义目标函数
def objective(x):
    return x**2 + 4*x + 4

result = optimize.minimize_scalar(objective)
print(f"函数 f(x) = x² + 4x + 4 的最小值:")
print(f"  最优 x: {result.x:.4f}")
print(f"  最小值: {result.fun:.4f}")

# 多元函数优化
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result2 = optimize.minimize(rosenbrock, x0=[0, 0])
print(f"\nRosenbrock 函数优化:")
print(f"  最优解: {result2.x}")
print(f"  最小值: {result2.fun:.6f}")

# 6. 曲线拟合
print("\n6. 曲线拟合")

# 生成带噪声的数据
x_data = np.linspace(0, 10, 50)
y_data = 2.5 * np.exp(-0.5 * x_data) + 1.5 + np.random.normal(0, 0.2, 50)

# 定义拟合函数
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = optimize.curve_fit(exp_func, x_data, y_data)
print(f"指数函数拟合 y = a*exp(-b*x) + c:")
print(f"  a = {popt[0]:.4f}")
print(f"  b = {popt[1]:.4f}")
print(f"  c = {popt[2]:.4f}")

# 7. 方程求根
print("\n7. 方程求根")

def equation(x):
    return x**2 - 4

root = optimize.root_scalar(equation, bracket=[0, 3])
print(f"方程 x² - 4 = 0 的根:")
print(f"  x = {root.root:.4f}")

# 8. 积分模块 (scipy.integrate)
print("\n8. 积分模块")

# 定积分
def integrand(x):
    return x**2

result, error = integrate.quad(integrand, 0, 3)
print(f"∫₀³ x² dx = {result:.4f} (误差: {error:.2e})")

# 二重积分
def integrand2(x, y):
    return np.exp(-x**2 - y**2)

result2, error2 = integrate.dblquad(integrand2, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
print(f"∬ exp(-x²-y²) dxdy = {result2:.4f} (理论值: π ≈ {np.pi:.4f})")

# 9. 插值模块 - 三次样条插值与可视化 (scipy.interpolate)
print("\n9. 插值模块")

x_points = np.array([0, 1, 2, 3, 4, 5])
y_points = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

linear_interp = interpolate.interp1d(x_points, y_points, kind='linear')
print(f"线性插值 x=2.5: {linear_interp(2.5):.4f}")

cubic_interp = interpolate.interp1d(x_points, y_points, kind='cubic')
print(f"三次样条插值 x=2.5: {cubic_interp(2.5):.4f}")

# 9.2 插值方法可视化对比
print("\n9.2 Interpolation methods comparison...")

x_pts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_pts = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0, -0.5, 0.3, 0.9, 0.2, -0.3])
x_fine = np.linspace(0, 10, 500)

li = interpolate.interp1d(x_pts, y_pts, kind='linear')
ci = interpolate.interp1d(x_pts, y_pts, kind='cubic')
ni = interpolate.interp1d(x_pts, y_pts, kind='nearest')
cs = interpolate.CubicSpline(x_pts, y_pts, bc_type='natural')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Interpolation Methods Comparison', fontsize=16, fontweight='bold')

axes[0, 0].plot(x_fine, li(x_fine), 'b-', linewidth=1.5, label='Linear')
axes[0, 0].plot(x_pts, y_pts, 'ro', markersize=8, label='Data points')
axes[0, 0].set_title('Linear Interpolation')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x_fine, ci(x_fine), 'g-', linewidth=1.5, label='Cubic spline')
axes[0, 1].plot(x_pts, y_pts, 'ro', markersize=8, label='Data points')
axes[0, 1].set_title('Cubic Spline Interpolation')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x_fine, ni(x_fine), 'm-', linewidth=1.5, label='Nearest')
axes[1, 0].plot(x_pts, y_pts, 'ro', markersize=8, label='Data points')
axes[1, 0].set_title('Nearest Neighbor Interpolation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x_fine, cs(x_fine), 'orange', linewidth=1.5, label='CubicSpline (natural)')
axes[1, 1].plot(x_pts, y_pts, 'ro', markersize=8, label='Data points')
axes[1, 1].set_title('CubicSpline Class')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to interpolation_comparison.png")
plt.close()

# 9.3 CubicSpline 求导数
print("\n9.3 CubicSpline derivatives...")

y_orig = cs(x_fine)
y_first_deriv = cs(x_fine, 1)
y_second_deriv = cs(x_fine, 2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(x_fine, y_orig, 'b-', linewidth=2, label='Cubic spline')
axes[0].plot(x_pts, y_pts, 'ro', markersize=8, label='Data points')
axes[0].set_title('Original Spline')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_fine, y_first_deriv, 'g-', linewidth=2, label="First derivative f'(x)")
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('First Derivative')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_fine, y_second_deriv, 'r-', linewidth=2, label="Second derivative f''(x)")
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].set_title('Second Derivative')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spline_derivatives.png', dpi=150, bbox_inches='tight')
print("Saved to spline_derivatives.png")
plt.close()

extrema_indices = np.where(np.diff(np.sign(y_first_deriv)))[0]
print("极值点:")
for idx in extrema_indices:
    x_ext = x_fine[idx]
    print(f"  x = {x_ext:.2f}, f(x) = {cs(x_ext):.4f}")

# 9.4 正弦函数采样与三次样条重建
print("\n9.4 Sine function reconstruction...")

x_true = np.linspace(0, 2 * np.pi, 1000)
y_true = np.sin(x_true)

x_sample = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
                     5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
y_sample = np.sin(x_sample)

cs_sin = interpolate.CubicSpline(x_sample, y_sample)
y_reconstructed = cs_sin(x_true)
error = np.abs(y_true - y_reconstructed)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(x_true, y_true, 'b-', linewidth=2, label='True sin(x)')
axes[0].plot(x_true, y_reconstructed, 'r--', linewidth=1.5, label='Cubic spline reconstruction')
axes[0].plot(x_sample, y_sample, 'ko', markersize=8, label='Sample points')
axes[0].set_title('sin(x) Reconstruction via Cubic Spline')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(x_true, 0, error, alpha=0.5, color='orange', label='|Error|')
axes[1].set_title(f'Reconstruction Error (max = {np.max(error):.6f})')
axes[1].set_xlabel('x')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spline_sine_reconstruction.png', dpi=150, bbox_inches='tight')
print("Saved to spline_sine_reconstruction.png")
plt.close()

# 9.5 二维插值
print("\n9.5 2D spline interpolation...")

from scipy.interpolate import RectBivariateSpline

x_2d = np.linspace(-3, 3, 11)
y_2d = np.linspace(-3, 3, 11)
X, Y = np.meshgrid(x_2d, y_2d)
Z = np.exp(-(X**2 + Y**2) / 2)

spline_2d = RectBivariateSpline(x_2d, y_2d, Z)

x_fine2 = np.linspace(-3, 3, 100)
y_fine2 = np.linspace(-3, 3, 100)
Z_fine = spline_2d(x_fine2, y_fine2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
axes[0].set_title('Original Data (11x11)')
plt.colorbar(im1, ax=axes[0])

X_f, Y_f = np.meshgrid(x_fine2, y_fine2)
im2 = axes[1].pcolormesh(X_f, Y_f, Z_fine, cmap='viridis', shading='auto')
axes[1].set_title('2D Spline Interpolation (100x100)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('spline_2d.png', dpi=150, bbox_inches='tight')
print("Saved to spline_2d.png")
plt.close()

# 9.6 B-spline 平滑
print("\n9.6 B-spline smoothing...")

x_bs = np.linspace(0, 10, 12)
y_bs = np.sin(x_bs) + np.random.normal(0, 0.1, 12)

tck = interpolate.splrep(x_bs, y_bs, s=0)
x_smooth = np.linspace(0, 10, 500)
y_smooth = interpolate.splev(x_smooth, tck)

plt.figure(figsize=(10, 5))
plt.plot(x_bs, y_bs, 'ro', markersize=8, label='Noisy data')
plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='B-spline fit')
plt.plot(x_smooth, np.sin(x_smooth), 'g--', linewidth=1.5, label='True sin(x)')
plt.title('B-spline Smoothing')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bspline_smoothing.png', dpi=150, bbox_inches='tight')
print("Saved to bspline_smoothing.png")
plt.close()

# 10. 线性代数 (scipy.linalg)
print("\n10. 线性代数")

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])

# 解线性方程组
x = linalg.solve(A, b)
print(f"解方程组 Ax = b:")
print(f"  A =\\n{A}")
print(f"  b = {b}")
print(f"  x = {x}")
print(f"  验证 Ax: {A @ x}")

# 矩阵分解
P, L, U = linalg.lu(A)
print(f"\nLU 分解:")
print(f"  P =\\n{P}")
print(f"  L =\\n{L}")
print(f"  U =\\n{U}")

# 特征值和特征向量
eigenvalues, eigenvectors = linalg.eig(A)
print(f"\n特征值: {eigenvalues}")
print(f"特征向量:\\n{eigenvectors}")

# 11. 空间算法 (scipy.spatial)
print("\n11. 空间算法")

points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
print(f"点集:\\n{points}")

# 计算距离矩阵
dist_matrix = spatial.distance_matrix(points, points)
print(f"\n距离矩阵:\\n{dist_matrix.round(2)}")

# 最近邻
query_point = np.array([[0.3, 0.3]])
dist, idx = spatial.KDTree(points).query(query_point)
print(f"\n点 (0.3, 0.3) 的最近邻:")
print(f"  距离: {dist}")
print(f"  索引: {idx} -> {points[idx]}")

# 12. 信号处理相关
print("\n12. 统计分布的可视化数据")

# 生成正态分布样本并检验正态性
np.random.seed(42)
normal_data = np.random.normal(0, 1, 100)
stat, p = stats.normaltest(normal_data)
print(f"正态性检验:")
print(f"  统计量: {stat:.4f}")
print(f"  p 值: {p:.4f}")
print(f"  结论: {'数据服从正态分布' if p > 0.05 else '数据不服从正态分布'}")

# 相关性检验
x_corr = np.array([1, 2, 3, 4, 5])
y_corr = np.array([2, 4, 5, 4, 5])
pearson_r, pearson_p = stats.pearsonr(x_corr, y_corr)
spearman_r, spearman_p = stats.spearmanr(x_corr, y_corr)
print(f"\n相关性分析:")
print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.4f}")
print(f"  Spearman r = {spearman_r:.4f}, p = {spearman_p:.4f}")

print("\n=== SciPy 基础操作完成 ===")
