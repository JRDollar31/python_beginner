"""
Advanced Transformer Topics

This script demonstrates:
1. RoPE (Rotary Position Embedding) with visualization
2. Flash Attention concept (block-wise computation)
3. KV Cache memory savings calculation
4. GQA (Grouped Query Attention) vs MHA vs MQA
5. SwiGLU activation vs ReLU/GELU in FFN
6. Architecture comparison: Transformer 2017 vs LLaMA-3 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

torch.manual_seed(42)

# ============================================================
# Part 1: RoPE (Rotary Position Embedding)
# ============================================================
print("=" * 60)
print("Part 1: RoPE - Rotary Position Embedding")
print("=" * 60)

print("""
RoPE Formula:

For position m and dimension pair (2i, 2i+1):
  q_m = R(m * theta) @ q
  k_n = R(n * theta) @ k

where R(theta) is a 2D rotation matrix:
  R(theta) = [cos(theta)  -sin(theta)]
             [sin(theta)   cos(theta)]

and theta_i = 10000^(-2i/d).

Key property: dot product of rotated q_m and k_n depends
only on relative position (m - n), enabling the model to
naturally capture relative position information.
""")


def apply_rope(q, k, base=10000):
    """
    Apply Rotary Position Embedding to query and key tensors.
    q, k: (batch, seq_len, n_heads, head_dim)
    """
    batch, seq_len, n_heads, head_dim = q.shape

    # Compute rotation angles for each dimension pair
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim//2)

    # Apply rotation: for each pair (2i, 2i+1), rotate by angle freqs[pos, i]
    def rotate_tensor(x):
        # x: (batch, seq_len, n_heads, head_dim)
        x_real = x[..., ::2]  # (batch, seq_len, n_heads, head_dim//2)
        x_imag = x[..., 1::2]  # (batch, seq_len, n_heads, head_dim//2)
        cos_t = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        sin_t = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
        x_rot_real = x_real * cos_t - x_imag * sin_t
        x_rot_imag = x_real * sin_t + x_imag * cos_t
        # Interleave back
        x_out = torch.stack([x_rot_real, x_rot_imag], dim=-1)
        return x_out.flatten(-2)  # (batch, seq_len, n_heads, head_dim)

    q_rot = rotate_tensor(q)
    k_rot = rotate_tensor(k)
    return q_rot, k_rot


# Visualize RoPE
D_HEAD = 16
SEQ_LEN = 20
q_orig = torch.randn(1, SEQ_LEN, 1, D_HEAD)
k_orig = torch.randn(1, SEQ_LEN, 1, D_HEAD)

q_rot, k_rot = apply_rope(q_orig, k_orig)

# Show the rotation effect on a 2D slice
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original vs rotated query (first head, first 2 dims)
ax = axes[0, 0]
positions = range(SEQ_LEN)
ax.plot(positions, q_orig[0, :, 0, 0].numpy(), "b-o", label="Original Q (dim 0)", markersize=4)
ax.plot(positions, q_rot[0, :, 0, 0].numpy(), "r--o", label="Rotated Q (dim 0)", markersize=4)
ax.plot(positions, q_orig[0, :, 0, 1].numpy(), "b-s", label="Original Q (dim 1)", markersize=4)
ax.plot(positions, q_rot[0, :, 0, 1].numpy(), "r--s", label="Rotated Q (dim 1)", markersize=4)
ax.set_xlabel("Position")
ax.set_ylabel("Value")
ax.set_title("RoPE: Original vs Rotated Query")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Rotation angle visualization
ax2 = axes[0, 1]
freqs = 1.0 / (10000 ** (torch.arange(0, D_HEAD, 2).float() / D_HEAD))
for i in range(0, min(D_HEAD, 16), 2):
    angle = torch.arange(SEQ_LEN).float() * freqs[i // 2]
    ax2.plot(range(SEQ_LEN), angle.numpy(), label=f"dim pair {i},{i+1}", alpha=0.7, linewidth=1)
ax2.set_xlabel("Position")
ax2.set_ylabel("Rotation Angle (radians)")
ax2.set_title("RoPE: Rotation Angles by Dimension Pair")
ax2.legend(fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3)

# Plot 3: Attention score with RoPE (relative position dependency)
ax3 = axes[1, 0]
# Compute attention scores between position 0 and all positions
q0 = q_rot[0, 0, 0, :]  # (D_HEAD,)
all_k = k_rot[0, :, 0, :]  # (SEQ_LEN, D_HEAD)
scores_rope = torch.matmul(q0, all_k.T).numpy()  # (SEQ_LEN,)

q0_orig = q_orig[0, 0, 0, :]
all_k_orig = k_orig[0, :, 0, :]
scores_no_rope = torch.matmul(q0_orig, all_k_orig.T).numpy()

ax3.plot(range(SEQ_LEN), scores_rope, "r-o", label="With RoPE", markersize=5)
ax3.plot(range(SEQ_LEN), scores_no_rope, "b--o", label="Without RoPE", markersize=5, alpha=0.5)
ax3.set_xlabel("Key Position")
ax3.set_ylabel("Attention Score (Q @ K)")
ax3.set_title("Attention Score: Query at Pos 0 vs All Key Positions")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Relative position encoding effect
ax4 = axes[1, 1]
# Show that RoPE scores depend on relative position
ref_pos = 5
scores_matrix = np.zeros((SEQ_LEN, SEQ_LEN))
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        qi = q_rot[0, i, 0, :]
        kj = k_rot[0, j, 0, :]
        scores_matrix[i, j] = torch.dot(qi, kj).item()

# Show diagonal (constant relative position = 0)
diagonal = np.diag(scores_matrix)
ax4.plot(range(SEQ_LEN), diagonal, "g-o", label="Relative pos = 0 (diagonal)", markersize=4)

# Show offset diagonal (relative position = 1)
offset1 = np.diag(scores_matrix, 1)
ax4.plot(range(SEQ_LEN - 1), offset1, "b-o", label="Relative pos = 1", markersize=4)

# Show offset diagonal (relative position = -1)
offset_minus1 = np.diag(scores_matrix, -1)
ax4.plot(range(1, SEQ_LEN), offset_minus1, "r-o", label="Relative pos = -1", markersize=4)

ax4.set_xlabel("Position Index")
ax4.set_ylabel("Attention Score")
ax4.set_title("RoPE: Scores by Relative Position")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/23_advanced_transformer_part1.png", dpi=150, bbox_inches="tight")
print("Saved plot to 23_advanced_transformer_part1.png")

# ============================================================
# Part 2: Flash Attention Concept
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Flash Attention Concept (Block-wise Computation)")
print("=" * 60)

print("""
Flash Attention reduces memory from O(N^2) to O(N) by:

1. Splitting Q, K, V into blocks
2. Computing attention scores block-by-block
3. Using online softmax normalization to avoid storing
   the full N x N attention matrix

Standard attention memory: O(N^2 * num_heads) for the score matrix
Flash attention memory: O(N * block_size * num_heads)

This enables training with much longer sequences on the same hardware.
""")


def standard_attention_memory(N, num_heads, dtype_bytes=4):
    """Memory needed for standard attention score matrix."""
    # Q@K^T creates (N, N) matrix per head
    return N * N * num_heads * dtype_bytes


def flash_attention_memory(N, num_heads, block_size, dtype_bytes=4):
    """Approximate memory for flash attention."""
    # Only need block_size x block_size at a time
    return block_size * block_size * num_heads * dtype_bytes + N * num_heads * 32  # + O(N) for output stats


N_values = [256, 512, 1024, 2048, 4096]
num_heads = 8
block_size = 128

standard_mem = [standard_attention_memory(n, num_heads) / (1024 * 1024) for n in N_values]
flash_mem = [flash_attention_memory(n, num_heads, block_size) / (1024 * 1024) for n in N_values]

print("Memory comparison (in MB):")
print(f"  {'Seq Len':>8} | {'Standard':>10} | {'Flash Attn':>10} | {'Savings':>8}")
print(f"  {'-'*8}-|-{'-'*10}-|-{'-'*10}-|-{'-'*8}")
for n, s, f in zip(N_values, standard_mem, flash_mem):
    savings = (1 - f / s) * 100
    print(f"  {n:>8} | {s:>10.2f} | {f:>10.2f} | {savings:>7.1f}%")

# ============================================================
# Part 3: KV Cache Memory Savings
# ============================================================
print("\n" + "=" * 60)
print("Part 3: KV Cache Memory Savings")
print("=" * 60)

print("""
KV Cache in autoregressive decoding:

During generation, we can cache K and V values from previous tokens
to avoid recomputing them. This trades memory for computation.

Without KV cache: recompute all K, V for each new token
  Cost: O(context_len * d_model) per token

With KV cache: store K, V for all previous tokens
  Memory: 2 * context_len * num_layers * num_heads * head_dim
  Cost: O(d_model) per token (only compute for new token)
""")

context_len = 4096
num_layers = 32
num_heads = 32
head_dim = 128
dtype_bytes = 2  # FP16

kv_cache_size = 2 * context_len * num_layers * num_heads * head_dim * dtype_bytes
kv_cache_gb = kv_cache_size / (1024 ** 3)

print(f"KV Cache size for generation:")
print(f"  Context length:   {context_len}")
print(f"  Num layers:       {num_layers}")
print(f"  Num heads:        {num_heads}")
print(f"  Head dim:         {head_dim}")
print(f"  KV cache size:    {kv_cache_gb:.2f} GB (FP16)")
print()

# Show how cache grows with sequence length
seq_lengths = [64, 256, 1024, 4096, 8192, 16384]
cache_sizes = [2 * sl * num_layers * num_heads * head_dim * dtype_bytes / (1024 ** 3) for sl in seq_lengths]

print("KV Cache growth with sequence length:")
for sl, cs in zip(seq_lengths, cache_sizes):
    print(f"  Context: {sl:>6} tokens | Cache: {cs:.2f} GB")

# ============================================================
# Part 4: GQA (Grouped Query Attention)
# ============================================================
print("\n" + "=" * 60)
print("Part 4: GQA vs MHA vs MQA")
print("=" * 60)

print("""
Attention Variants:

1. MHA (Multi-Head Attention): Each query head has its own KV heads
   - Params: num_heads * (W_q + W_k + W_v)
   - Most expressive, most memory

2. MQA (Multi-Query Attention): All query heads share ONE KV head
   - Params: num_heads * W_q + 1 * (W_k + W_v)
   - Least memory, but quality drop

3. GQA (Grouped Query Attention): Query heads grouped into G groups,
   each group shares one KV head
   - Params: num_heads * W_q + num_groups * (W_k + W_v)
   - Best tradeoff: used in LLaMA-2/3, Mistral

KV cache savings:
  MHA: 2 * num_heads * head_dim per layer per token
  GQA: 2 * num_groups * head_dim per layer per token
  MQA: 2 * head_dim per layer per token
""")


def compute_attention_params(d_model, num_heads, num_kv_heads):
    """Compute parameter counts for attention variants."""
    # Q projection: always num_heads * head_dim = d_model
    q_params = d_model * d_model
    # K and V projections: num_kv_heads * head_dim each
    kv_dim = num_kv_heads * (d_model // num_heads)
    k_params = d_model * kv_dim
    v_params = d_model * kv_dim
    # Output projection
    o_params = d_model * d_model
    return q_params + k_params + v_params + o_params


D_MODEL = 4096
NUM_HEADS = 32

configs = {
    "MHA (32 heads)": {"num_heads": 32, "num_kv_heads": 32},
    "GQA (32Q, 8KV)": {"num_heads": 32, "num_kv_heads": 8},
    "GQA (32Q, 4KV)": {"num_heads": 32, "num_kv_heads": 4},
    "MQA (32Q, 1KV)": {"num_heads": 32, "num_kv_heads": 1},
}

print(f"\nAttention parameter comparison (d_model={D_MODEL}):")
print(f"  {'Config':<20} | {'Parameters':>12} | {'KV Cache / token':>16}")
print(f"  {'-'*20}-|-{'-'*12}-|-{'-'*16}")

param_counts = []
cache_sizes = []
labels = []

for name, cfg in configs.items():
    params = compute_attention_params(D_MODEL, cfg["num_heads"], cfg["num_kv_heads"])
    kv_cache_per_token = 2 * cfg["num_kv_heads"] * (D_MODEL // cfg["num_heads"]) * 2  # FP16
    param_counts.append(params / 1e6)
    cache_sizes.append(kv_cache_per_token / 1024)  # KB
    labels.append(name)
    print(f"  {name:<20} | {params:>12,} | {kv_cache_per_token:>12,} B")

# Plot comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes2[0]
bars = ax.bar(labels, param_counts, color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"])
ax.set_xlabel("Attention Type")
ax.set_ylabel("Parameters (millions)")
ax.set_title("Attention Parameter Count Comparison")
for bar, val in zip(bars, param_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}M",
            ha="center", fontweight="bold")

ax2 = axes2[1]
bars2 = ax2.bar(labels, cache_sizes, color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"])
ax2.set_xlabel("Attention Type")
ax2.set_ylabel("KV Cache per token (KB)")
ax2.set_title("KV Cache Size per Token per Layer")
for bar, val in zip(bars2, cache_sizes):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.0f}KB",
             ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/23_advanced_transformer_part2.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to 23_advanced_transformer_part2.png")

# ============================================================
# Part 5: SwiGLU vs ReLU/GELU FFN
# ============================================================
print("\n" + "=" * 60)
print("Part 5: SwiGLU vs ReLU vs GELU in FFN")
print("=" * 60)

print("""
FFN Activation Comparison:

Standard FFN:     FFN(x) = (x @ W1) activated @ W2
SwiGLU FFN:       FFN(x) = (x @ W1) * SiLU(x @ W3) @ W2

where SiLU(x) = x * sigmoid(x) (also called Swish)

SwiGLU uses a gating mechanism: one branch is linearly projected
and gated by the SiLU-activated other branch.

Benefits:
- Better gradient flow (gating prevents saturation)
- Used in LLaMA, PaLM, and other modern models
- Slightly more parameters (3 projections vs 2) but better quality
""")


def swish(x):
    return x * torch.sigmoid(x)


def swiglu(x, w1, w2, w3):
    """SwiGLU: (x @ W1) * SiLU(x @ W3) @ W2"""
    return (x @ w1) * swish(x @ w3) @ w2


def relu_ffn(x, w1, w2):
    """Standard ReLU FFN: ReLU(x @ W1) @ W2"""
    return F.relu(x @ w1) @ w2


def gelu_ffn(x, w1, w2):
    """GELU FFN: GELU(x @ W1) @ W2"""
    return F.gelu(x @ w1) @ w2


# Compare gradient flow
D_IN = 64
D_FF = 256
D_OUT = 64

torch.manual_seed(42)
x = torch.randn(100, D_IN, requires_grad=True)

# SwiGLU
w1_s = torch.randn(D_IN, D_FF) / math.sqrt(D_IN)
w2_s = torch.randn(D_FF, D_OUT) / math.sqrt(D_FF)
w3_s = torch.randn(D_IN, D_FF) / math.sqrt(D_IN)
x_s = x.clone().detach().requires_grad_(True)
out_s = swiglu(x_s, w1_s, w2_s, w3_s).mean()
out_s.backward()
swiglu_grad_norm = x_s.grad.norm().item()

# ReLU
w1_r = torch.randn(D_IN, D_FF) / math.sqrt(D_IN)
w2_r = torch.randn(D_FF, D_OUT) / math.sqrt(D_FF)
x_r = x.clone().detach().requires_grad_(True)
out_r = relu_ffn(x_r, w1_r, w2_r).mean()
out_r.backward()
relu_grad_norm = x_r.grad.norm().item()

# GELU
w1_g = torch.randn(D_IN, D_FF) / math.sqrt(D_IN)
w2_g = torch.randn(D_FF, D_OUT) / math.sqrt(D_FF)
x_g = x.clone().detach().requires_grad_(True)
out_g = gelu_ffn(x_g, w1_g, w2_g).mean()
out_g.backward()
gelu_grad_norm = x_g.grad.norm().item()

print(f"Gradient norm comparison (same input, random weights):")
print(f"  ReLU:    {relu_grad_norm:.4f}")
print(f"  GELU:    {gelu_grad_norm:.4f}")
print(f"  SwiGLU:  {swiglu_grad_norm:.4f}")

# Plot activation function shapes
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes3[0]
x_range = torch.linspace(-3, 3, 200)
ax.plot(x_range.numpy(), F.relu(x_range).numpy(), label="ReLU", linewidth=2)
ax.plot(x_range.numpy(), F.gelu(x_range).numpy(), label="GELU", linewidth=2)
ax.plot(x_range.numpy(), swish(x_range).numpy(), label="SiLU (Swish)", linewidth=2)
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.set_title("Activation Functions")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)

# Gradient flow comparison
ax2 = axes3[1]
activations = ["ReLU", "GELU", "SwiGLU"]
grad_norms = [relu_grad_norm, gelu_grad_norm, swiglu_grad_norm]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
bars = ax2.bar(activations, grad_norms, color=colors)
ax2.set_xlabel("Activation Function")
ax2.set_ylabel("Gradient Norm")
ax2.set_title("Gradient Flow Comparison")
for bar, val in zip(bars, grad_norms):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
             ha="center", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/23_advanced_transformer_part3.png", dpi=150, bbox_inches="tight")
print("Saved plot to 23_advanced_transformer_part3.png")

# ============================================================
# Part 6: Architecture Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("Part 6: Architecture Comparison - Transformer 2017 vs LLaMA-3 2024")
print("=" * 60)

# Print formatted table
header = f"{'Feature':<30} | {'Transformer (2017)':<20} | {'LLaMA-3 8B (2024)':<20}"
print(header)
print("-" * len(header))

rows = [
    ("Parameters", "~65M (base)", "8B"),
    ("Layers", "6 enc + 6 dec", "32 decoder-only"),
    ("d_model", "512", "4096"),
    ("Attention heads", "8", "32"),
    ("d_kv (head dim)", "64", "128"),
    ("FFN dim", "2048", "14336"),
    ("Seq length", "512", "8192"),
    ("Position encoding", "Absolute (learned)", "RoPE"),
    ("Attention type", "MHA", "GQA (8 KV groups)"),
    ("Activation", "ReLU", "SwiGLU"),
    ("Normalization", "Post-norm", "Pre-norm (RMSNorm)"),
    ("Bias in projections", "Yes", "No"),
    ("Training data", "WMT EN-DE (~4.5M)", "~15T tokens"),
    ("Vocabulary size", "37K (BPE)", "128K (TikToken)"),
]

for feature, t2017, llama3 in rows:
    print(f"{feature:<30} | {t2017:<20} | {llama3:<20}")

print("\n" + "=" * 60)
print("Key Evolution Summary")
print("=" * 60)
print("1. Decoder-only architecture (simpler, good for generation)")
print("2. RoPE for position encoding (better length extrapolation)")
print("3. GQA for efficient KV cache (faster inference)")
print("4. SwiGLU activation (better gradient flow)")
print("5. Pre-normalization (more stable training)")
print("6. RMSNorm (simpler, faster than LayerNorm)")
print("7. Massive scale increase: 65M -> 8B parameters")
print("8. Much longer context: 512 -> 8192 tokens")
