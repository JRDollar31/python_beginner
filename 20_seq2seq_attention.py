"""
Seq2Seq and Attention Demo

This script demonstrates:
1. Encoder-Decoder architecture with GRU
2. Training on a sequence reversal task
3. Attention mechanism visualization
4. Training loss and accuracy curves
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

# ============================================================
# Configuration
# ============================================================
VOCAB_SIZE = 20
SEQ_LEN = 6
HIDDEN_DIM = 64
EMBED_DIM = 32
NUM_EPOCHS = 50
BATCH_SIZE = 128
BATCHES_PER_EPOCH = 20
LEARNING_RATE = 0.005
torch.manual_seed(42)

device = torch.device("cpu")
print("=" * 60)
print("Seq2Seq with Attention - Sequence Reversal Task")
print("=" * 60)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Hidden dimension: {HIDDEN_DIM}")
print(f"Embedding dimension: {EMBED_DIM}")
print(f"Device: {device}")

# ============================================================
# Data Generation
# ============================================================

def generate_batch(batch_size=BATCH_SIZE):
    """Generate a batch of random sequences and their reversed versions."""
    src = torch.randint(1, VOCAB_SIZE, (batch_size, SEQ_LEN))
    tgt = torch.flip(src, dims=[1])
    # Decoder input: prepend SOS token (0), remove last element
    decoder_input = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long), tgt[:, :-1]], dim=1)
    return src, tgt, decoder_input


# ============================================================
# Attention Mechanism
# ============================================================

class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim)
        src_len = encoder_outputs.size(1)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden_expanded, encoder_outputs], dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1)  # (batch, src_len)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        return context.squeeze(1), attention_weights


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)  # (batch, seq_len, embed_dim)
        outputs, hidden = self.gru(embedded)  # outputs: (batch, seq_len, hidden)
        return outputs, hidden


# ============================================================
# Decoder with Attention
# ============================================================

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, decoder_input, hidden, encoder_outputs):
        embedded = self.embedding(decoder_input)  # (batch, 1, embed_dim)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        prediction = self.fc(torch.cat([output, context], dim=1))
        return prediction, hidden, attn_weights


# ============================================================
# Seq2Seq Model
# ============================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, seq_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len

    def forward(self, src, decoder_input):
        encoder_outputs, hidden = self.encoder(src)
        # hidden shape: (1, batch, hidden_dim)
        outputs = []
        all_attn_weights = []

        for t in range(self.seq_len):
            pred, hidden, attn_weights = self.decoder(
                decoder_input[:, t:t+1], hidden.squeeze(0), encoder_outputs
            )
            # hidden returned as (1, batch, hidden_dim) from GRU
            outputs.append(pred)
            all_attn_weights.append(attn_weights)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab)
        all_attn_weights = torch.stack(all_attn_weights, dim=1)  # (batch, seq_len, src_len)
        return outputs, all_attn_weights


# ============================================================
# Training
# ============================================================

print("\nInitializing model...")
encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
decoder = DecoderWithAttention(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
model = Seq2Seq(encoder, decoder, SEQ_LEN).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accs = []

print("Training...")
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for _ in range(BATCHES_PER_EPOCH):
        src, tgt, dec_input = generate_batch()
        src, tgt, dec_input = src.to(device), tgt.to(device), dec_input.to(device)

        optimizer.zero_grad()
        outputs, _ = model(src, dec_input)

        # outputs: (batch, seq_len, vocab), tgt: (batch, seq_len)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=2)
        epoch_correct += (preds == tgt).sum().item()
        epoch_total += tgt.numel()

    avg_loss = epoch_loss / BATCHES_PER_EPOCH
    accuracy = epoch_correct / epoch_total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# ============================================================
# Inference Demo
# ============================================================

print("\n" + "-" * 40)
print("Inference Examples:")
print("-" * 40)

model.eval()
with torch.no_grad():
    src, tgt, _ = generate_batch(batch_size=5)
    _, attn_weights = model(src, torch.zeros(5, SEQ_LEN, dtype=torch.long))

    for i in range(5):
        src_seq = src[i].tolist()
        tgt_seq = tgt[i].tolist()
        print(f"  Input:  {src_seq}")
        print(f"  Target: {tgt_seq}")

# ============================================================
# Plot Attention Weights Heatmap
# ============================================================

print("\nPlotting attention weights...")

# Use attention weights from a single sample
sample_attn = attn_weights[0].numpy()  # (seq_len, src_len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Attention heatmap
ax = axes[0]
im = ax.imshow(sample_attn, cmap="Blues", aspect="auto")
ax.set_xlabel("Source Position")
ax.set_ylabel("Decoder Step")
ax.set_title("Attention Weights (Sequence Reversal)")
ax.set_xticks(range(SEQ_LEN))
ax.set_yticks(range(SEQ_LEN))
ax.set_xticklabels([str(i) for i in range(SEQ_LEN)])
ax.set_yticklabels([str(i) for i in range(SEQ_LEN)])

# Add text annotations
for ii in range(SEQ_LEN):
    for jj in range(SEQ_LEN):
        ax.text(jj, ii, f"{sample_attn[ii, jj]:.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if sample_attn[ii, jj] < 0.5 else "white")

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Training curves
ax2 = axes[1]
epochs = range(1, NUM_EPOCHS + 1)
ax2.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss", color="b")
ax2.set_title("Training Loss Curve")
ax2.grid(True, alpha=0.3)

# Add accuracy on secondary axis
ax3 = ax2.twinx()
ax3.plot(epochs, train_accs, "g-", label="Accuracy", linewidth=2)
ax3.set_ylabel("Accuracy", color="g")
ax3.set_ylim(0, 1.1)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/20_seq2seq_attention_results.png", dpi=150, bbox_inches="tight")
print("Saved plot to 20_seq2seq_attention_results.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final training accuracy: {train_accs[-1]:.4f}")
print("The attention heatmap shows how the decoder attends to different")
print("source positions at each decoding step. For sequence reversal,")
print("we expect the attention to focus on opposite positions.")
print("For example, decoder step 0 should attend to source position 5.")
