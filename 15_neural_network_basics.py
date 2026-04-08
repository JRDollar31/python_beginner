# 15_neural_network_basics.py
# Neural Network Basics - From Perceptron to Gradient Descent

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
print("15 Neural Network Basics - Perceptron to Gradient Descent")
print("=" * 60)

# === 1. Perceptron ===
print("\n=== 1. Perceptron Calculation ===")
x = torch.tensor([2.0, 3.0])
w = torch.tensor([0.5, -0.2])
b = torch.tensor(0.1)
z = torch.dot(w, x) + b
print(f"Input x = {x}")
print(f"Weight w = {w}")
print(f"Bias b = {b}")
print(f"Output z = w.x + b = {z:.4f}")

# === 2. Activation Functions ===
print("\n=== 2. Activation Functions ===")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

test_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"{'x':<8} {'Sigmoid':>10} {'Tanh':>10} {'ReLU':>10} {'GELU':>10}")
print("-" * 52)
for v in test_values:
    print(f"{v.item():<8.2f} {sigmoid(v).item():>10.4f} {torch.tanh(v).item():>10.4f} {relu(v).item():>10.4f} {gelu(v).item():>10.4f}")

# Visualize activation functions
x_range = torch.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].plot(x_range.numpy(), sigmoid(x_range).numpy(), 'b-', linewidth=2)
axes[0].set_title('Sigmoid')
axes[0].grid(True, alpha=0.3)
axes[1].plot(x_range.numpy(), torch.tanh(x_range).numpy(), 'g-', linewidth=2)
axes[1].set_title('Tanh')
axes[1].grid(True, alpha=0.3)
axes[2].plot(x_range.numpy(), relu(x_range).numpy(), 'r-', linewidth=2)
axes[2].set_title('ReLU')
axes[2].grid(True, alpha=0.3)
axes[3].plot(x_range.numpy(), gelu(x_range).numpy(), 'm-', linewidth=2)
axes[3].set_title('GELU')
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('15_activation_functions.png', dpi=100)
plt.close()
print("\nActivation functions plot saved: 15_activation_functions.png")

# === 3. Loss Functions ===
print("\n=== 3. Loss Function (MSE) Example ===")
targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
pred_good = torch.tensor([1.1, 1.9, 3.1, 3.9])
pred_bad = torch.tensor([2.0, 0.5, 5.0, 2.0])
mse_good = torch.mean((pred_good - targets)**2)
mse_bad = torch.mean((pred_bad - targets)**2)
print(f"Good prediction:  MSE = {mse_good:.4f}")
print(f"Bad prediction:   MSE = {mse_bad:.4f}")

# === 4. 1D Gradient Descent ===
print("\n=== 4. 1D Gradient Descent ===")

def f(x):
    return x**2 + 2*x + 3

def df(x):
    return 2*x + 2

x = 3.0
lr = 0.1
print(f"{'Step':<6} {'x':>10} {'f(x)':>10} {'Grad':>10}")
print("-" * 40)
for step in range(10):
    grad = df(x)
    print(f"{step:<6} {x:>10.4f} {f(x):>10.4f} {grad:>10.4f}")
    x = x - lr * grad

print(f"\nAfter 10 steps: x = {x:.4f}, f(x) = {f(x):.4f}")
print(f"Theoretical minimum: x = -1.0, f(x) = 2.0")

# === 5. Train a Simple Neural Network ===
print("\n=== 5. Training a Neural Network with PyTorch ===")

X_train = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
y_train = torch.sin(X_train) + 0.1 * torch.randn(200, 1)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = SimpleNN()
print(f"Model:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_history = []

print(f"\n{'Epoch':<8} {'Loss':>12}")
print("-" * 25)

for epoch in range(500):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"{epoch+1:<8} {loss.item():>12.6f}")

# Visualize training results
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(loss_history, 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss Curve')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

model.eval()
with torch.no_grad():
    y_pred = model(X_train)

axes[1].scatter(X_train.numpy(), y_train.numpy(), s=10, alpha=0.3, label='Data')
axes[1].plot(X_train.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='Prediction')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Fitting Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

weights = []
for name, param in model.named_parameters():
    if 'weight' in name:
        weights.extend(param.detach().numpy().flatten())
axes[2].hist(weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Weight Value')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Weight Distribution')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_training_result.png', dpi=100)
plt.close()
print("\nTraining result plot saved: 15_training_result.png")

with torch.no_grad():
    y_pred = model(X_train)
    ss_res = torch.sum((y_train - y_pred)**2)
    ss_tot = torch.sum((y_train - y_train.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    print(f"\nR-squared: {r_squared.item():.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")

print("\n=== Summary ===")
print("1. Perceptron: z = w.x + b")
print("2. Activation: non-linearity (Sigmoid, Tanh, ReLU, GELU)")
print("3. Loss: measures prediction error (MSE, Cross-Entropy)")
print("4. Gradient Descent: theta = theta - eta * grad(L)")
print("5. Training: forward -> loss -> backward -> update")
