# 18_rnn_fundamentals.py
# RNN Fundamentals - Forward Pass, BPTT, Vanishing Gradients

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
print("18 RNN Fundamentals - Forward Pass and BPTT")
print("=" * 60)

# === 1. RNN Forward Pass (Manual) ===
print("\n=== 1. RNN Forward Pass (Manual Calculation) ===")

input_size = 3
hidden_size = 2

W_ih = np.array([[0.1, 0.2, 0.3], [-0.2, 0.8, 0.6]])
b_ih = np.array([0.1, -0.1])
W_hh = np.array([[0.7, 0.8], [0.9, 1.0]])
b_hh = np.array([0.0, 0.1])

sequence = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

print(f"Input sequence (seq_len=4, input_size=3):")
for t, x in enumerate(sequence):
    print(f"  t={t}: {x}")

h = np.zeros(hidden_size)
print(f"\n{'t':<4} {'Input x_t':<20} {'W_ih.x':<18} {'W_hh.h':<18} {'z':<18} {'h_t=tanh(z)':>14}")
print("-" * 100)

hidden_states = [h.copy()]
for t, x in enumerate(sequence):
    input_part = W_ih @ x + b_ih
    hidden_part = W_hh @ h + b_hh
    z = input_part + hidden_part
    h = np.tanh(z)
    hidden_states.append(h.copy())
    print(f"{t:<4} {str(x):<20} {str(np.round(input_part,4)):<18} {str(np.round(hidden_part,4)):<18} {str(np.round(z,4)):<18} {str(np.round(h,4)):>14}")

print(f"\nFinal hidden state: h_4 = {np.round(h, 4)}")

# === 2. Verify with PyTorch ===
print("\n=== 2. Verify with PyTorch nn.RNN ===")

rnn = nn.RNN(input_size=3, hidden_size=2, batch_first=True)
with torch.no_grad():
    rnn.weight_ih_l0.copy_(torch.from_numpy(W_ih))
    rnn.bias_ih_l0.copy_(torch.from_numpy(b_ih))
    rnn.weight_hh_l0.copy_(torch.from_numpy(W_hh))
    rnn.bias_hh_l0.copy_(torch.from_numpy(b_hh))

x_torch = torch.from_numpy(sequence).unsqueeze(0).float()
output, h_n = rnn(x_torch)

print(f"PyTorch final hidden state: {h_n[0].detach().numpy()}")
print(f"Manual final hidden state:  {np.round(h, 4)}")
print(f"Match: {np.allclose(h, h_n[0].detach().numpy(), atol=1e-5)}")

# === 3. Gradient Vanishing in RNN ===
print("\n=== 3. Gradient Vanishing Analysis in RNN ===")

tanh_avg_deriv = 0.7
W_scales = [0.3, 0.5, 0.8, 1.0, 1.2]
seq_lengths = [5, 10, 20, 50, 100]

print("Gradient decay factor = (W_scale * tanh')^T")
print(f"{'W_scale':<12}", end='')
for sl in seq_lengths:
    print(f"{'T='+str(sl):<12}", end='')
print()
print("-" * 72)

for ws in W_scales:
    decay_factor = ws * tanh_avg_deriv
    print(f"{ws:<12.1f}", end='')
    for sl in seq_lengths:
        decay = decay_factor ** sl
        print(f"{decay:<12.2e}", end='')
    print()

print("\nConclusion: With W_scale=0.3, after 20 steps gradient decays to ~0,")
print("making it nearly impossible for RNN to learn long-range dependencies.")

# === 4. RNN Sequence Prediction (Sine Wave) ===
print("\n=== 4. RNN Sine Wave Prediction ===")

torch.manual_seed(42)

def generate_sine_data(seq_length=30, n_samples=500):
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.uniform(0, 4 * math.pi)
        x_vals = np.linspace(start, start + seq_length * 0.2, seq_length + 1)
        sine_vals = np.sin(x_vals)
        X.append(sine_vals[:seq_length].reshape(-1, 1))
        y.append(sine_vals[seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

seq_len = 30
X_train, y_train = generate_sine_data(seq_len, 500)
X_test, y_test = generate_sine_data(seq_len, 100)

class SineRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

model = SineRNN(hidden_size=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

losses = []
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"Training complete. Final loss: {losses[-1]:.6f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(losses, 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

axes[1].scatter(y_test.numpy(), y_pred.numpy(), s=10, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
r2 = 1 - torch.sum((y_test - y_pred)**2) / torch.sum((y_test - y_test.mean())**2)
axes[1].set_xlabel('True Value')
axes[1].set_ylabel('Predicted Value')
axes[1].set_title(f'Prediction vs True (R2={r2:.4f})')
axes[1].grid(True, alpha=0.3)

sample_idx = 0
sample_input = X_test[sample_idx].numpy().flatten()
sample_pred = y_pred[sample_idx].item()
sample_true = y_test[sample_idx].item()
axes[2].plot(range(seq_len), sample_input, 'b-o', markersize=4, linewidth=1.5, label='Input')
axes[2].plot(seq_len, sample_true, 'g*', markersize=15, label=f'True={sample_true:.3f}')
axes[2].plot(seq_len, sample_pred, 'r^', markersize=15, label=f'Pred={sample_pred:.3f}')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('sin value')
axes[2].set_title('Single Sample Prediction')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('18_rnn_sine_prediction.png', dpi=100)
plt.close()
print("RNN prediction plot saved: 18_rnn_sine_prediction.png")

print("\n=== Summary ===")
print("1. RNN: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)")
print("2. Same parameters shared across all time steps")
print("3. BPTT: unroll RNN through time, then apply backprop")
print("4. Gradient vanishing: RNN struggles with sequences > 10 steps")
print("5. RNN works well for short-range prediction (sine wave)")
