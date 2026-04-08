"""
Transformer Architecture - Building Blocks from Scratch

This script demonstrates:
1. MultiHeadAttention implemented from scratch
2. PositionalEncoding with sin/cos functions
3. Attention weights visualization
4. Gradient flow analysis with different initializations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

torch.manual_seed(42)

# ============================================================
# Part 1: MultiHeadAttention from Scratch
# ============================================================
print("=" * 60)
print("Part 1: MultiHeadAttention from Scratch")
print("=" * 60)

print("""
MultiHeadAttention Formula:

1. For each head i:
   Q_i = X @ W_Q_i    (batch, seq_len, d_k)
   K_i = X @ W_K_i    (batch, seq_len, d_k)
   V_i = X @ W_V_i    (batch, seq_len, d_v)

2. Scaled dot-product attention:
   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

3. Concatenate all heads:
   MultiHead = Concat(head_1, ..., head_h) @ W_O

Key insight: Multiple heads allow the model to attend to
different positions and representations simultaneously.
""")


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implemented from scratch."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query)  # (batch, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # (batch, heads, seq, seq)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)  # (batch, heads, seq, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output, attention_weights


# Test MultiHeadAttention
D_MODEL = 64
NUM_HEADS = 4
SEQ_LEN = 10
BATCH_SIZE = 2

mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
output, attn_weights = mha(x, x, x)

print(f"MultiHeadAttention test:")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Attention weights shape: {attn_weights.shape}")
print(f"  Number of heads: {NUM_HEADS}")
print(f"  Dimension per head (d_k): {D_MODEL // NUM_HEADS}")

# ============================================================
# Part 2: Positional Encoding (Sin/Cos)
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Positional Encoding (Sin/Cos)")
print("=" * 60)

print("""
Positional Encoding Formula:

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why sin/cos?
- Allows the model to attend to relative positions
  (PE(pos+k) can be represented as linear function of PE(pos))
- Provides unique encoding for each position
- Bounded values prevent explosion for long sequences
""")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


pe = PositionalEncoding(D_MODEL, max_len=100)

# Visualize positional encoding
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Positional encoding for first few dimensions
ax = axes[0, 0]
positions = np.arange(50)
for dim in [0, 1, 2, 3]:
    ax.plot(positions, pe.pe[0, :50, dim].numpy(), label=f"dim {dim}")
ax.set_xlabel("Position")
ax.set_ylabel("PE Value")
ax.set_title("Positional Encoding (First 4 Dimensions)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Heatmap of positional encoding
ax2 = axes[0, 1]
pe_matrix = pe.pe[0, :50, :].numpy().T  # (d_model, seq_len)
im = ax2.imshow(pe_matrix, cmap="RdYlBu", aspect="auto")
ax2.set_xlabel("Position")
ax2.set_ylabel("Dimension")
ax2.set_title("Positional Encoding Heatmap")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# ============================================================
# Part 3: Attention Weights Visualization
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Attention Weights Visualization")
print("=" * 60)

# Create sample input with some structure
sample_input = torch.zeros(1, SEQ_LEN, D_MODEL)
# Add distinct patterns at different positions
for pos in range(SEQ_LEN):
    sample_input[0, pos, pos % D_MODEL] = 1.0

_, sample_attn = mha(sample_input, sample_input, sample_input)
# Average over heads for visualization
avg_attn = sample_attn.mean(dim=1).squeeze(0).detach().numpy()  # (seq_len, seq_len)

# Plot attention weights
ax3 = axes[1, 0]
im3 = ax3.imshow(avg_attn, cmap="Blues", aspect="auto")
ax3.set_xlabel("Key Position")
ax3.set_ylabel("Query Position")
ax3.set_title("Average Attention Weights (All Heads)")
ax3.set_xticks(range(SEQ_LEN))
ax3.set_yticks(range(SEQ_LEN))
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# Show per-head attention
ax4 = axes[1, 1]
# Plot attention from position 0 across all heads
head_attn = sample_attn.squeeze(0)[:, 0, :].detach().numpy()  # (num_heads, seq_len)
x_pos = np.arange(SEQ_LEN)
width = 0.8 / NUM_HEADS
for h in range(NUM_HEADS):
    ax4.bar(x_pos + h * width, head_attn[h], width, label=f"Head {h}")
ax4.set_xlabel("Key Position")
ax4.set_ylabel("Attention Weight")
ax4.set_title("Attention from Position 0 (Per Head)")
ax4.set_xticks(x_pos + width * (NUM_HEADS - 1) / 2)
ax4.set_xticklabels([str(i) for i in range(SEQ_LEN)])
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/21_transformer_arch_part1.png", dpi=150, bbox_inches="tight")
print("Saved plot to 21_transformer_arch_part1.png")

# ============================================================
# Part 4: Gradient Flow with Different Initializations
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Gradient Flow vs Initialization")
print("=" * 60)

print("""
Testing how different initialization strategies affect
gradient flow through the MultiHeadAttention module.
""")


def measure_gradient_flow(init_method, model):
    """Measure gradient norms after a backward pass."""
    # Apply initialization
    if init_method == "default":
        pass  # Use default PyTorch init
    elif init_method == "xavier":
        for name, param in model.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    elif init_method == "kaiming":
        for name, param in model.named_parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param)
    elif init_method == "small":
        for name, param in model.named_parameters():
            if param.dim() > 1:
                nn.init.normal_(param, std=0.01)

    # Forward + backward
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=False)
    model.zero_grad()
    output, _ = model(x, x, x)
    loss = output.pow(2).mean()
    loss.backward()

    # Collect gradient norms per layer
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            short_name = name.split(".")[-1]
            if short_name not in grad_norms:
                grad_norms[short_name] = []
            grad_norms[short_name].append(param.grad.norm().item())

    # Average gradient norms per parameter type
    result = {k: np.mean(v) for k, v in grad_norms.items()}
    return result


init_methods = ["default", "xavier", "kaiming", "small"]
all_grad_norms = {}

for method in init_methods:
    test_model = MultiHeadAttention(D_MODEL, NUM_HEADS)
    grad_norms = measure_gradient_flow(method, test_model)
    all_grad_norms[method] = grad_norms
    print(f"  {method:8s} | w_o grad norm: {grad_norms.get('w_o', 0):.6f}")

# Plot gradient flow comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of gradient norms
param_types = ["w_q", "w_k", "w_v", "w_o"]
x_pos2 = np.arange(len(param_types))
width2 = 0.2

for i, method in enumerate(init_methods):
    norms = [all_grad_norms[method].get(p, 0) for p in param_types]
    axes2[0].bar(x_pos2 + i * width2, norms, width2, label=method)

axes2[0].set_xlabel("Parameter")
axes2[0].set_ylabel("Gradient Norm")
axes2[0].set_title("Gradient Norms by Initialization Method")
axes2[0].set_xticks(x_pos2 + width2 * 1.5)
axes2[0].set_xticklabels(param_types)
axes2[0].legend()
if all(v > 0 for v in [xavier_norms, kaiming_norms, small_norms, large_norms]):
    axes2[0].set_yscale("log")
axes2[0].grid(True, alpha=0.3, axis="y")

# Layer-wise gradient flow for xavier init
ax2_2 = axes2[1]
xavier_norms = [all_grad_norms["xavier"].get(p, 0) for p in param_types]
ax2_2.plot(param_types, xavier_norms, "o-", linewidth=2, markersize=8)
ax2_2.set_xlabel("Layer")
ax2_2.set_ylabel("Gradient Norm")
ax2_2.set_title("Xavier Initialization: Gradient Flow")
if all(v > 0 for v in xavier_norms):
    ax2_2.set_yscale("log")
ax2_2.grid(True, alpha=0.3)

# Add annotation
for j, (p, v) in enumerate(zip(param_types, xavier_norms)):
    ax2_2.annotate(f"{v:.2e}", (p, v), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/21_transformer_arch_part2.png", dpi=150, bbox_inches="tight")
print("Saved plot to 21_transformer_arch_part2.png")

# ============================================================
# Part 5: Architecture Summary
# ============================================================
print("\n" + "=" * 60)
print("Transformer Architecture Summary")
print("=" * 60)

total_params = sum(p.numel() for p in mha.parameters())
print(f"MultiHeadAttention parameters: {total_params:,}")
print(f"  - d_model: {D_MODEL}")
print(f"  - num_heads: {NUM_HEADS}")
print(f"  - d_k per head: {D_MODEL // NUM_HEADS}")
print()
print("Key design choices:")
print("  1. Scaled dot-product: divide by sqrt(d_k) to prevent")
print("     softmax saturation for large d_k")
print("  2. Multiple heads: each head learns different aspects")
print("  3. Sin/cos positional encoding: enables relative position")
print("     reasoning without learned parameters")
print("  4. Xavier initialization provides stable gradient flow")
