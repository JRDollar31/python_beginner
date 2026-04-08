# 17_backpropagation_deep_dive.py
# Backpropagation - Chain Rule and Numerical Examples

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 60)
print("17 Backpropagation - Chain Rule and Numerical Examples")
print("=" * 60)

# === 1. Chain Rule Demo ===
print("\n=== 1. Chain Rule Numerical Demo ===")

x = 3.0; w = 2.0; b = 1.0; target = 10.0
y = w * x + b
L = (y - target)**2

print(f"Forward:")
print(f"  x={x}, w={w}, b={b}")
print(f"  y = w*x+b = {y}")
print(f"  L = (y-target)^2 = {L}")

dL_dy = 2 * (y - target)
dy_dw = x
dy_db = 1.0
dL_dw = dL_dy * dy_dw
dL_db = dL_dy * dy_db

print(f"\nBackward (chain rule):")
print(f"  dL/dy = 2*(y-target) = {dL_dy}")
print(f"  dL/dw = dL/dy * dy/dw = {dL_dy} * {x} = {dL_dw}")
print(f"  dL/db = dL/dy * dy/db = {dL_dy} * 1 = {dL_db}")

# Verify with PyTorch autograd
x_t = torch.tensor(3.0)
w_t = torch.tensor(2.0, requires_grad=True)
b_t = torch.tensor(1.0, requires_grad=True)
y_t = w_t * x_t + b_t
L_t = (y_t - torch.tensor(10.0))**2
L_t.backward()

print(f"\nPyTorch autograd verification:")
print(f"  dL/dw = {w_t.grad.item()} (manual: {dL_dw})")
print(f"  dL/db = {b_t.grad.item()} (manual: {dL_db})")

# === 2. Full Network Backprop ===
print("\n=== 2. Full Network Backpropagation ===")

W1 = np.array([[0.5, -0.3], [-0.2, 0.8]])
b1 = np.array([0.1, -0.1])
W2 = np.array([0.4, 0.6])
b2 = 0.2
x = np.array([1.0, 0.5])
y_true = 0.8

# Forward
z1 = W1 @ x + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = W2 @ a1 + b2
y_pred = z2
loss = 0.5 * (y_pred - y_true)**2

print(f"Forward pass:")
print(f"  z1 = {np.round(z1, 4)}")
print(f"  a1 = sigmoid(z1) = {np.round(a1, 4)}")
print(f"  z2 (prediction) = {z2:.6f}")
print(f"  Loss = {loss:.6f}")

# Backward
dz2 = y_pred - y_true
dW2 = dz2 * a1
db2 = dz2
da1 = dz2 * W2
dz1 = da1 * a1 * (1 - a1)
dW1 = np.outer(dz1, x)
db1 = dz1

print(f"\nBackward pass:")
print(f"  dW2 = {np.round(dW2, 4)}")
print(f"  db2 = {db2:.6f}")
print(f"  dW1 =\n{np.round(dW1, 4)}")
print(f"  db1 = {np.round(db1, 4)}")

# === 3. Gradient Vanishing Demo ===
print("\n=== 3. Gradient Vanishing Demo ===")

num_layers = 10
sigmoid_max_grad = 0.25
weight_scale = 0.5
grad_output = 1.0

gradients = [grad_output]
for _ in range(num_layers):
    gradients.append(gradients[-1] * sigmoid_max_grad * weight_scale)

print(f"{'Layer':<8} {'Gradient':>15} {'Decay Factor':>12}")
print("-" * 40)
for i, g in enumerate(gradients):
    print(f"L{num_layers-i:<5} {g:>15.2e} {g/gradients[0]:>12.2e}")

print(f"\nAfter {num_layers} layers, gradient decays to {gradients[-1]/gradients[0]:.2e} of original")

# Visualize gradient decay
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(gradients)), gradients, color='steelblue', alpha=0.7)
ax.set_xlabel('Layer (from output to input)')
ax.set_ylabel('Gradient Magnitude')
ax.set_title(f'Gradient Vanishing ({num_layers} layers)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('17_gradient_vanishing.png', dpi=100)
plt.close()
print("Gradient vanishing plot saved: 17_gradient_vanishing.png")

# === 4. Gradient Explosion Demo ===
print("\n=== 4. Gradient Explosion Demo ===")

weight_scale_large = 3.0
grad_output_small = 0.01
gradients_exp = [grad_output_small]
for _ in range(num_layers):
    gradients_exp.append(gradients_exp[-1] * weight_scale_large)

print(f"{'Layer':<8} {'Gradient':>15} {'Amplification':>12}")
print("-" * 40)
for i, g in enumerate(gradients_exp):
    print(f"L{num_layers-i:<5} {g:>15.2e} {g/gradients_exp[0]:>12.2e}")

# === 5. Initialization Comparison ===
print("\n=== 5. Weight Initialization Effect on Gradients ===")

def test_initialization(init_method, n_layers=6, dim=10):
    torch.manual_seed(42)
    layers = []
    for i in range(n_layers):
        layer = nn.Linear(dim, dim)
        if init_method == 'xavier':
            nn.init.xavier_normal_(layer.weight)
        elif init_method == 'he':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif init_method == 'large':
            nn.init.normal_(layer.weight, std=3.0)
        elif init_method == 'small':
            nn.init.normal_(layer.weight, std=0.01)
        layers.append(layer)
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    
    x = torch.randn(1, dim, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    layer_grads = []
    for name_p, param in model.named_parameters():
        if 'weight' in name_p and param.grad is not None:
            layer_grads.append(param.grad.norm().item())
    return layer_grads

methods = [('Xavier', 'xavier'), ('He (ReLU)', 'he'), ('Small (std=0.01)', 'small'), ('Large (std=3.0)', 'large')]

print(f"{'Method':<20} {'Min Grad':>12} {'Max Grad':>12} {'Ratio':>10}")
print("-" * 58)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, method) in enumerate(methods):
    grads = test_initialization(method)
    ratio = max(grads) / (min(grads) + 1e-10)
    print(f"{name:<20} {min(grads):>12.6f} {max(grads):>12.6f} {ratio:>10.2f}")
    
    ax = axes[idx]
    ax.bar(range(len(grads)), grads, color='steelblue', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'{name}: [{min(grads):.6f}, {max(grads):.6f}]')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('17_initialization_comparison.png', dpi=100)
plt.close()
print("\nInitialization comparison saved: 17_initialization_comparison.png")

print("\n=== Summary ===")
print("1. Chain rule: dL/dw = dL/dz * dz/dw")
print("2. Backprop applies chain rule layer by layer")
print("3. Gradient vanishing: small weights/sigmoid -> gradients decay to 0")
print("4. Gradient explosion: large weights -> gradients grow exponentially")
print("5. Proper initialization (Xavier/He) keeps gradients stable")
