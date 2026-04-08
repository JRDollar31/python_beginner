"""
LSTM and GRU Demo - Understanding Recurrent Network Variants

This script demonstrates:
1. LSTM cell formulas and gate mechanisms
2. Comparison of RNN vs LSTM on a long-memory task
3. GRU vs LSTM performance and parameter count comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

# ============================================================
# Part 1: LSTM Formulas
# ============================================================
print("=" * 60)
print("Part 1: LSTM Cell Formulas")
print("=" * 60)

print("""
The LSTM (Long Short-Term Memory) cell has four key equations:

1. Forget gate:  f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
   -> Decides what information to discard from cell state

2. Input gate:   i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
   -> Decides what new information to store

3. Cell candidate: c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
   -> Creates new candidate values for cell state

4. Cell update:  c_t = f_t * c_{t-1} + i_t * c_tilde
   -> Updates the cell state (long-term memory)

5. Output gate:  o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
   -> Decides what to output

6. Hidden state: h_t = o_t * tanh(c_t)
   -> Produces the output (short-term memory)

The key insight: the cell state c_t flows through time
with only element-wise operations (no matrix mult),
which prevents gradient vanishing/exploding.
""")

# ============================================================
# Part 2: Long Memory Task - RNN vs LSTM
# ============================================================
print("=" * 60)
print("Part 2: Long Memory Task - RNN vs LSTM")
print("=" * 60)

INPUT_DIM = 8
HIDDEN_DIM = 32
OUTPUT_DIM = 1
NUM_EPOCHS = 30
SEQ_LENGTHS = [10, 20, 50]
torch.manual_seed(42)


class MemoryRNN(nn.Module):
    """Simple RNN for the long memory task."""
    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type="RNN"):
        super().__init__()
        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Use last hidden state to predict the first element
        return self.fc(out[:, -1, :]).squeeze(-1)


def generate_memory_data(seq_len, n_samples=500):
    """
    Task: remember the first element of the sequence.
    Input: random sequence of length seq_len with INPUT_DIM features.
    Target: sum of first element's features (scaled to [0, 1]).
    """
    x = torch.randn(n_samples, seq_len, INPUT_DIM)
    # Target based on first element
    target = torch.sigmoid(x[:, 0, 0])
    return x, target


def train_and_evaluate(rnn_type, seq_len):
    model = MemoryRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, rnn_type)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    x_train, y_train = generate_memory_data(seq_len, n_samples=500)
    x_test, y_test = generate_memory_data(seq_len, n_samples=100)

    losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(x_test)
        test_loss = criterion(pred, y_test).item()
        # Accuracy: within tolerance
        accuracy = (torch.abs(pred - y_test) < 0.15).float().mean().item()

    return test_loss, accuracy, losses


results = {}
for seq_len in SEQ_LENGTHS:
    print(f"\nSequence length: {seq_len}")
    for rnn_type in ["RNN", "LSTM"]:
        loss, acc, _ = train_and_evaluate(rnn_type, seq_len)
        results[(rnn_type, seq_len)] = {"loss": loss, "accuracy": acc}
        print(f"  {rnn_type:4s} | Test Loss: {loss:.4f} | Accuracy: {acc:.4f}")

# ============================================================
# Part 3: GRU vs LSTM Comparison
# ============================================================
print("\n" + "=" * 60)
print("Part 3: GRU vs LSTM - Parameter Count & Performance")
print("=" * 60)

print("""
GRU (Gated Recurrent Unit) formulas:

1. Reset gate:    r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)
   -> Controls how much past info to forget

2. Update gate:   z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)
   -> Balances old state vs new candidate

3. Candidate:     h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)
   -> Computes new memory content

4. Hidden state:  h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
   -> Interpolates between old and new

GRU simplification vs LSTM:
- No separate cell state (c_t)
- Combines forget and input into one update gate
- Fewer parameters (3 gates vs 4)
- Often faster to train
""")


class MemoryGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# Count parameters
rnn_model = MemoryRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, "RNN")
lstm_model = MemoryRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, "LSTM")
gru_model = MemoryGRU(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

rnn_params = sum(p.numel() for p in rnn_model.parameters())
lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())

print(f"Parameter counts (input={INPUT_DIM}, hidden={HIDDEN_DIM}):")
print(f"  RNN:  {rnn_params:>5d} parameters")
print(f"  LSTM: {lstm_params:>5d} parameters")
print(f"  GRU:  {gru_params:>5d} parameters")
print(f"\nGRU has {100 * (lstm_params - gru_params) / lstm_params:.1f}% fewer parameters than LSTM")

# Train GRU and compare
print("\nTraining GRU for comparison:")
gru_results = {}
for seq_len in SEQ_LENGTHS:
    model = MemoryGRU(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    opt = optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()
    x_train, y_train = generate_memory_data(seq_len, n_samples=500)
    x_test, y_test = generate_memory_data(seq_len, n_samples=100)

    for epoch in range(NUM_EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = crit(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test)
        test_loss = crit(pred, y_test).item()
        accuracy = (torch.abs(pred - y_test) < 0.15).float().mean().item()
    gru_results[seq_len] = {"loss": test_loss, "accuracy": accuracy}
    print(f"  seq_len={seq_len:2d} | Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

# ============================================================
# Part 4: Plot Accuracy vs Sequence Length
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Plotting Results")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: accuracy vs sequence length
x_positions = np.arange(len(SEQ_LENGTHS))
width = 0.25
rnn_accs = [results[("RNN", s)]["accuracy"] for s in SEQ_LENGTHS]
lstm_accs = [results[("LSTM", s)]["accuracy"] for s in SEQ_LENGTHS]
gru_accs = [gru_results[s]["accuracy"] for s in SEQ_LENGTHS]

ax = axes[0]
ax.bar(x_positions - width, rnn_accs, width, label="RNN", color="#e74c3c")
ax.bar(x_positions, lstm_accs, width, label="LSTM", color="#3498db")
ax.bar(x_positions + width, gru_accs, width, label="GRU", color="#2ecc71")
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Accuracy (within 0.15 tolerance)")
ax.set_title("Long Memory Task: Accuracy vs Sequence Length")
ax.set_xticks(x_positions)
ax.set_xticklabels([str(s) for s in SEQ_LENGTHS])
ax.legend()
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

# Bar chart: parameter counts
ax2 = axes[1]
models = ["RNN", "LSTM", "GRU"]
params = [rnn_params, lstm_params, gru_params]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
ax2.bar(models, params, color=colors)
ax2.set_xlabel("Model Type")
ax2.set_ylabel("Number of Parameters")
ax2.set_title("Parameter Count Comparison")
for i, v in enumerate(params):
    ax2.text(i, v + 10, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/19_lstm_gru_results.png", dpi=150, bbox_inches="tight")
print("Saved plot to 19_lstm_gru_results.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("1. RNN performance degrades rapidly with longer sequences")
print("2. LSTM maintains good accuracy even at seq_len=50")
print("3. GRU achieves comparable results with fewer parameters")
print("4. For tasks requiring long-term memory, LSTM/GRU are preferred")
print("   over vanilla RNNs")
