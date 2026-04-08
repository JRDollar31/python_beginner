# 16_optimization_algorithms.py
# Optimization Algorithms: SGD, Momentum, Adam

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 60)
print("16 Optimization Algorithms - SGD, Momentum, Adam")
print("=" * 60)

# === 1. Gradient Descent Variants ===
print("\n=== 1. Gradient Descent Variants ===")

def valley_loss(w1, w2):
    return 0.1 * w1**2 + 2.0 * w2**2

def grad_valley(w1, w2):
    return 0.2 * w1, 4.0 * w2

def run_sgd(x0, y0, lr, epochs):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(epochs):
        dx, dy = grad_valley(x, y)
        x -= lr * dx
        y -= lr * dy
        path.append((x, y))
    return np.array(path)

def run_momentum(x0, y0, lr, beta, epochs):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    path = [(x, y)]
    for _ in range(epochs):
        dx, dy = grad_valley(x, y)
        vx = beta * vx + (1 - beta) * dx
        vy = beta * vy + (1 - beta) * dy
        x -= lr * vx
        y -= lr * vy
        path.append((x, y))
    return np.array(path)

def run_rmsprop(x0, y0, lr, gamma, epochs):
    x, y = x0, y0
    eg2_x, eg2_y = 0.0, 0.0
    path = [(x, y)]
    for _ in range(epochs):
        dx, dy = grad_valley(x, y)
        eg2_x = gamma * eg2_x + (1 - gamma) * dx**2
        eg2_y = gamma * eg2_y + (1 - gamma) * dy**2
        x -= lr * dx / (np.sqrt(eg2_x) + 1e-8)
        y -= lr * dy / (np.sqrt(eg2_y) + 1e-8)
        path.append((x, y))
    return np.array(path)

def run_adam(x0, y0, lr, beta1, beta2, eps, epochs):
    x, y = x0, y0
    m_x, m_y = 0.0, 0.0
    v_x, v_y = 0.0, 0.0
    path = [(x, y)]
    for t in range(1, epochs + 1):
        dx, dy = grad_valley(x, y)
        m_x = beta1 * m_x + (1 - beta1) * dx
        m_y = beta1 * m_y + (1 - beta1) * dy
        v_x = beta2 * v_x + (1 - beta2) * dx**2
        v_y = beta2 * v_y + (1 - beta2) * dy**2
        m_hat_x = m_x / (1 - beta1**t)
        m_hat_y = m_y / (1 - beta1**t)
        v_hat_x = v_x / (1 - beta2**t)
        v_hat_y = v_y / (1 - beta2**t)
        x -= lr * m_hat_x / (np.sqrt(v_hat_x) + eps)
        y -= lr * m_hat_y / (np.sqrt(v_hat_y) + eps)
        path.append((x, y))
    return np.array(path)

# Visualize optimization paths
x_r = np.linspace(-3, 3, 100)
y_r = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_r, y_r)
Z = valley_loss(X, Y)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

optimizers = [
    ('SGD', lambda: run_sgd(2.5, 2.5, lr=0.15, epochs=50), 'blue'),
    ('Momentum', lambda: run_momentum(2.5, 2.5, lr=0.5, beta=0.9, epochs=50), 'green'),
    ('RMSProp', lambda: run_rmsprop(2.5, 2.5, lr=0.1, gamma=0.9, epochs=50), 'orange'),
    ('Adam', lambda: run_adam(2.5, 2.5, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, epochs=50), 'red'),
]

for idx, (name, run_fn, color) in enumerate(optimizers):
    ax = axes[idx]
    ax.contourf(X, Y, Z, levels=30, cmap='coolwarm')
    path = run_fn()
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, markersize=4, marker='o', alpha=0.7)
    ax.plot([0], [0], 'r*', markersize=12)
    final_loss = valley_loss(path[-1, 0], path[-1, 1])
    ax.set_title(f'{name}\nFinal: ({path[-1,0]:.4f}, {path[-1,1]:.4f})\nLoss: {final_loss:.6f}')

plt.tight_layout()
plt.savefig('16_optimizer_comparison.png', dpi=100)
plt.close()
print("Optimizer comparison plot saved: 16_optimizer_comparison.png")

# === 2. Compare optimizers in real neural network ===
print("\n=== 2. Optimizer Comparison in Neural Network ===")

X = torch.linspace(-2, 2, 100).unsqueeze(1)
y_true = X**2
y = y_true + 0.3 * torch.randn(100, 1)

class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

optimizers_config = [
    ('SGD', lambda params: optim.SGD(params, lr=0.05), 'blue'),
    ('SGD+Momentum', lambda params: optim.SGD(params, lr=0.05, momentum=0.9), 'green'),
    ('Adam', lambda params: optim.Adam(params, lr=0.01), 'red'),
    ('RMSprop', lambda params: optim.RMSprop(params, lr=0.01), 'orange'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

print(f"{'Optimizer':<15} {'Final MSE':>12}")
print("-" * 30)

for idx, (name, opt_fn, color) in enumerate(optimizers_config):
    torch.manual_seed(42)
    model = SmallNN()
    optimizer = opt_fn(model.parameters())
    criterion = nn.MSELoss()
    loss_hist = []
    for epoch in range(200):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
    
    ax = axes[idx]
    ax.plot(loss_hist, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{name}\nFinal Loss: {loss_hist[-1]:.6f}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    print(f"{name:<15} {loss_hist[-1]:>12.6f}")

plt.tight_layout()
plt.savefig('16_nn_optimizer_comparison.png', dpi=100)
plt.close()
print("\nNN optimizer comparison saved: 16_nn_optimizer_comparison.png")

# === 3. Learning Rate Schedulers ===
print("\n=== 3. Learning Rate Schedulers ===")

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

schedulers_config = [
    ('Fixed LR', None, 'blue'),
    ('StepLR (x0.5/50)', 'step', 'green'),
    ('Cosine Annealing', 'cosine', 'orange'),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, (name, sched_type, color) in enumerate(schedulers_config):
    torch.manual_seed(42)
    model = SmallNN()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    if sched_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    elif sched_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.001)
    else:
        scheduler = None
    
    criterion = nn.MSELoss()
    loss_hist = []
    lr_hist = []
    
    for epoch in range(200):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        lr_hist.append(optimizer.param_groups[0]['lr'])
        if scheduler:
            scheduler.step()
    
    ax = axes[idx]
    ax.plot(loss_hist, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{name}\nFinal Loss: {loss_hist[-1]:.6f}')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(lr_hist, 'gray', linewidth=1, linestyle='--')
    ax2.set_ylabel('Learning Rate', color='gray')

plt.tight_layout()
plt.savefig('16_lr_schedulers.png', dpi=100)
plt.close()
print("LR scheduler plot saved: 16_lr_schedulers.png")

print("\n=== Summary ===")
print("1. SGD: simple, oscillates in narrow valleys")
print("2. Momentum: accumulates velocity, faster convergence")
print("3. RMSProp: adaptive per-parameter learning rate")
print("4. Adam = Momentum + RMSProp, best default choice")
print("5. LR schedulers: StepLR, CosineAnnealing improve results")
