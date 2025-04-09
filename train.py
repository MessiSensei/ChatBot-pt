import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import json
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score
from tokenizer import SimpleTokenizer
from load_data import load_dataset
from model import EncoderRNN, Attention, DecoderRNN, Seq2Seq

# Config
EMBED_SIZE = 64
HIDDEN_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0005
MODEL_NAME = "model2.pt"
PLOT_DIR = "static"
DATASET_FILE = "data/dataset.txt"
MODEL_DIR = "models"
VOCAB_FILE = "vocab.txt"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Loading data...")
input_texts, target_texts = load_dataset(DATASET_FILE)
tokenizer = SimpleTokenizer()
START_TOKEN, END_TOKEN, PAD_TOKEN = "<start>", "<end>", "<pad>"
input_texts = [t.lower() for t in input_texts]
target_texts = [f"{START_TOKEN} {t.lower()} {END_TOKEN}" for t in target_texts]

tokenizer.build_vocab(input_texts + target_texts)
tokenizer.save_vocab(VOCAB_FILE)

vocab_size = len(tokenizer.word2idx)

input_seqs = [torch.tensor(tokenizer.encode(t)) for t in input_texts]
target_seqs = [torch.tensor(tokenizer.encode(t)) for t in target_texts]
split = int(0.9 * len(input_seqs))
train_input, val_input = input_seqs[:split], input_seqs[split:]
train_target, val_target = target_seqs[:split], target_seqs[split:]

combined = list(zip(train_input, train_target))
random.shuffle(combined)
train_input[:], train_target[:] = zip(*combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

encoder = EncoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, dropout=0.3)
attn = Attention(HIDDEN_SIZE)
decoder = DecoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, attn, dropout=0.3)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses, val_accuracies = [], [], []

plt.ion()
fig, ax = plt.subplots()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for i in range(0, len(train_input), BATCH_SIZE):
        inputs = torch.nn.utils.rnn.pad_sequence(train_input[i:i+BATCH_SIZE], batch_first=True,
                                                 padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
        targets = torch.nn.utils.rnn.pad_sequence(train_target[i:i+BATCH_SIZE], batch_first=True,
                                                  padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
        optimizer.zero_grad()
        output = model(inputs, targets)
        loss = criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_input)
    train_losses.append(avg_train_loss)

    # === Validation ===
    model.eval()
    val_loss, correct_tokens, total_tokens = 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(val_input), BATCH_SIZE):
            inputs = torch.nn.utils.rnn.pad_sequence(val_input[i:i+BATCH_SIZE], batch_first=True,
                                                     padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
            targets = torch.nn.utils.rnn.pad_sequence(val_target[i:i+BATCH_SIZE], batch_first=True,
                                                      padding_value=tokenizer.word2idx[PAD_TOKEN]).to(device)
            output = model(inputs, targets, teacher_forcing_ratio=0.0)
            preds = output.argmax(dim=2)

            mask = targets != tokenizer.word2idx[PAD_TOKEN]
            correct_tokens += (preds == targets).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            val_loss += criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1)).item()

    avg_val_loss = val_loss / len(val_input)
    val_losses.append(avg_val_loss)

    val_acc = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    ax.clear()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.plot(val_accuracies, label='Val Acc')
    ax.legend()
    ax.set_title("Training Metrics")
    plt.pause(0.1)

# Save model
final_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
torch.save(model.state_dict(), final_model_path)
print("Model saved to:", final_model_path)

plt.ioff()
plt.savefig(os.path.join(PLOT_DIR, "training_plot.png"))
