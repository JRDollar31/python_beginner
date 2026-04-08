"""
Transformer Training - Copy Task

This script demonstrates:
1. Training a Transformer on a sequence copy task
2. Using nn.TransformerEncoder + nn.TransformerDecoder
3. Training loss curve, accuracy curve, and learning rate schedule
4. Greedy decoding inference with example outputs
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
# Configuration
# ============================================================
VOCAB_SIZE = 20
SEQ_LEN = 8
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FF = 128
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

device = torch.device("cpu")
print("=" * 60)
print("Transformer Training - Sequence Copy Task")
print("=" * 60)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"d_model: {D_MODEL}, nhead: {NHEAD}")
print(f"Encoder layers: {NUM_ENCODER_LAYERS}, Decoder layers: {NUM_DECODER_LAYERS}")
print(f"Device: {device}")

# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Transformer Model
# ============================================================

class CopyTransformer(nn.Module):
    """Transformer for sequence-to-sequence copy task."""

    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

    def forward(self, src, tgt):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        src_embed = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_embed = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        memory = self.transformer_encoder(src_embed)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)

        logits = self.fc_out(output)
        return logits

    def greedy_decode(self, src, max_len):
        """Greedy decoding for inference."""
        self.eval()
        batch_size = src.size(0)
        src_embed = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_embed)

        # Start with SOS token
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=src.device) * SOS_TOKEN
        outputs = []

        for _ in range(max_len):
            tgt_embed = self.pos_encoder(self.embedding(decoder_input) * math.sqrt(self.d_model))
            tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(src.device)
            output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            outputs.append(next_token)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        return torch.cat(outputs, dim=1)  # (batch, max_len)


# ============================================================
# Data Generation
# ============================================================

def generate_copy_batch(batch_size=BATCH_SIZE):
    """Generate random sequences; target is identical to source."""
    # Content tokens start from 3 (0=PAD, 1=SOS, 2=EOS)
    src = torch.randint(3, VOCAB_SIZE, (batch_size, SEQ_LEN))
    tgt = src.clone()
    # Decoder input: prepend SOS
    dec_input = torch.cat([torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long), tgt[:, :-1]], dim=1)
    return src, tgt, dec_input


# ============================================================
# Learning Rate Schedule
# ============================================================

class WarmupLR:
    """Warmup then inverse sqrt decay."""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        lr = min(lr, 0.01)  # Cap learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ============================================================
# Training Loop
# ============================================================

print("\nInitializing model...")
model = CopyTransformer(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FF
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = WarmupLR(optimizer, D_MODEL, warmup_steps=2000)

train_losses = []
train_accs = []
lr_history = []

print("Training...")
batches_per_epoch = 30

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for _ in range(batches_per_epoch):
        src, tgt, dec_input = generate_copy_batch()
        src, tgt, dec_input = src.to(device), tgt.to(device), dec_input.to(device)

        optimizer.zero_grad()
        logits = model(src, dec_input)

        # logits: (batch, tgt_len, vocab), tgt: (batch, tgt_len)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current_lr = scheduler.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=-1)
        epoch_correct += (preds == tgt).sum().item()
        epoch_total += tgt.numel()

    avg_loss = epoch_loss / batches_per_epoch
    accuracy = epoch_correct / epoch_total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    lr_history.append(current_lr)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | LR: {current_lr:.6f}")

# ============================================================
# Inference Demo
# ============================================================

print("\n" + "-" * 40)
print("Greedy Decoding Inference Examples:")
print("-" * 40)

model.eval()
with torch.no_grad():
    src_examples = torch.randint(3, VOCAB_SIZE, (5, SEQ_LEN))
    predictions = model.greedy_decode(src_examples, SEQ_LEN)

    for i in range(5):
        src_seq = src_examples[i].tolist()
        pred_seq = predictions[i].tolist()
        match = "CORRECT" if src_seq == pred_seq else "MISMATCH"
        print(f"  Input:    {src_seq}")
        print(f"  Predicted:{pred_seq}  [{match}]")

# ============================================================
# Plotting
# ============================================================

print("\nPlotting training curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs = range(1, NUM_EPOCHS + 1)

# Training loss
ax = axes[0]
ax.plot(epochs, train_losses, "b-", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")
ax.grid(True, alpha=0.3)

# Training accuracy
ax2 = axes[1]
ax2.plot(epochs, train_accs, "g-", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy Curve")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

# Learning rate schedule
ax3 = axes[2]
ax3.plot(epochs, lr_history, "r-", linewidth=2)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("Learning Rate Schedule (Warmup + Decay)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/root/ubuntu/python_beginner/22_transformer_training_results.png", dpi=150, bbox_inches="tight")
print("Saved plot to 22_transformer_training_results.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final training accuracy: {train_accs[-1]:.4f}")
print(f"Total model parameters: {total_params:,}")
print()
print("The copy task tests whether the transformer can learn")
print("the identity function. With sufficient training, the")
print("model should achieve near-perfect accuracy.")
print()
print("The learning rate schedule uses warmup (linear increase)")
print("followed by inverse square root decay, which is the")
print("standard schedule from the original Transformer paper.")
