"""
Author : MessiSensei

Usage: educational purpose 
"""
from flask import Flask, render_template, request, jsonify, send_file, Response
import torch
import threading
import glob, os, time, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model import EncoderRNN, Attention, DecoderRNN, Seq2Seq , beam_search_decode
from tokenizer import SimpleTokenizer
from load_data import load_dataset
import json
import re
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time  
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score ,confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
        
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("static"):
    os.makedirs("static")

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer & Model Initialization
tokenizer = SimpleTokenizer()
if os.path.exists("vocab.txt"):
    tokenizer.load_vocab("vocab.txt")
else:
    print(" vocab.txt not found — tokenizer will be built during training.")

vocab_size = len(tokenizer.word2idx)
model = None
model_lock = threading.Lock()
training_thread = None
selected_model_file = "models/model1.pt"


log_file_path = "logs/train.log"
log_messages = []
stop_training_flag = False

# Load Pretrained Model 
def load_model(filename):
    global model

    # Ensure vocab is available
    vocab_path = "vocab.txt"
    if not os.path.exists(vocab_path):
        raise RuntimeError(" vocab.txt not found. Please clean your dataset and build vocabulary before loading a model.")

    # Load vocab
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.word2idx)
    if vocab_size == 0:
        raise RuntimeError("vocab.txt is empty. Cannot initialize model with vocab_size = 0.")

    # Load model config
    config_path = filename.replace(".pt", ".json")
    if not os.path.exists(config_path):
        raise RuntimeError(" Missing config file for this model.")

    with open(config_path, "r") as f:
        config = json.load(f)

    embed_size = config.get("embed_size", 32)
    hidden_size = config.get("hidden_size", 64)

    # Build model components
    encoder = EncoderRNN(vocab_size, embed_size, hidden_size, dropout=0.3)
    attention = Attention(hidden_size)
    decoder = DecoderRNN(vocab_size, embed_size, hidden_size, attention, dropout=0.3)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load model weights
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    print(f" Loaded model from {filename} with vocab size {vocab_size}, embed_size {embed_size}, hidden_size {hidden_size}")



# Chatbot Response Function
def generate_response(
    input_text,
    max_len=30,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    use_beam=True,
    beam_width=3
):
    input_tokens = tokenizer.encode(input_text.lower())
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with model_lock:
        if use_beam:
            # Beam Search Decoding 
            beam_output = beam_search_decode(model, tokenizer, input_tensor, beam_width=beam_width)
            return beam_output

        else:
            #Top-k / Top-p Sampling Decoding 
            encoder_outputs, hidden = model.encoder(input_tensor)
            input_token = torch.tensor([tokenizer.word2idx[START_TOKEN]], dtype=torch.long).to(device)

            output_ids = []
            attention_weights = []

            for _ in range(max_len):
                output, hidden, attn_weights = model.decoder(
                    input_token, hidden, encoder_outputs, return_attention=True
                )
                logits = output.squeeze(0) / temperature
                probs = torch.softmax(logits, dim=-1)

                if top_k > 0:
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(0, topk_indices, topk_probs)
                    probs /= probs.sum()
                elif top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative_probs > top_p
                    cutoff_idx = torch.where(cutoff)[0][0] + 1
                    probs = torch.zeros_like(probs)
                    probs[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
                    probs /= probs.sum()

                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token in [tokenizer.word2idx[END_TOKEN], tokenizer.word2idx[PAD_TOKEN]]:
                    break

                if next_token not in [tokenizer.word2idx[START_TOKEN], tokenizer.word2idx[END_TOKEN]]:
                    output_ids.append(next_token)

                attention_weights.append(attn_weights.cpu().detach().numpy())
                input_token = torch.tensor([next_token], dtype=torch.long).to(device)

            input_words = [tokenizer.idx2word.get(idx, "<unk>") for idx in input_tokens]
            output_words = [tokenizer.idx2word.get(idx, "<unk>") for idx in output_ids]

            def filter_special_tokens(tokens):
                return [t for t in tokens if t not in ["<pad>", "<unk>", "<start>", "<end>"]]

            filtered_input = filter_special_tokens(input_words)
            filtered_output = filter_special_tokens(output_words)

            if attention_weights:
                attention_tensor = torch.cat([torch.tensor(w) for w in attention_weights], dim=0)
                attention_matrix = attention_tensor.squeeze(1).cpu().numpy()
                attention_matrix = attention_matrix[:len(filtered_output), :len(filtered_input)]
                save_attention_heatmap(attention_matrix, filtered_input, filtered_output)
            else:
                plt.figure()
                plt.text(0.5, 0.5, "No attention weights", ha="center", va="center")
                plt.axis("off")
                plt.savefig("static/attention_heatmap.png")
                plt.close()

            return tokenizer.decode(output_ids)


def save_attention_heatmap(attention_matrix, input_tokens, output_tokens):
    # These should already be decoded word strings — no need to decode again
    input_labels = input_tokens
    output_labels = output_tokens

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(attention_matrix, aspect='auto', cmap='viridis')

    ax.set_xticks(np.arange(len(input_labels)))
    ax.set_yticks(np.arange(len(output_labels)))
    ax.set_xticklabels(input_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(output_labels, fontsize=9)
    ax.set_title("Attention Heatmap")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("static/attention_heatmap.png")
    plt.close()



def log(msg):
    log_messages.append(msg)
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")
        
#  Training Logic 
def train_model(params):
    global model, selected_model_file, stop_training_flag
    stop_training_flag = False
    top_k_tokens = params.get("top_k_tokens", 30)  # default = 30
    input_texts, target_texts = load_dataset("data/dataset.txt")
    input_texts = [t.lower() for t in input_texts]
    target_texts = [f"{START_TOKEN} {t.lower()} {END_TOKEN}" for t in target_texts]
    teacher_forcing_ratio = params.get("teacher_forcing_ratio", 1.0)

    #  build vocab
    tokenizer.build_vocab(input_texts + target_texts)
    tokenizer.save_vocab("vocab.txt")
    assert "<pad>" in tokenizer.word2idx, " <pad> missing!"
    assert "<unk>" in tokenizer.word2idx, " <unk> missing!"
    assert "<start>" in tokenizer.word2idx, " <start> missing!"
    assert "<end>" in tokenizer.word2idx, " <end> missing!"

    vocab_size = len(tokenizer.word2idx)
    log(f" Vocabulary built: {vocab_size} tokens.")
    log(f" Vocab built with {len(tokenizer.word2idx)} tokens.")


    input_seqs = [torch.tensor(tokenizer.encode(t)) for t in input_texts]
    target_seqs = [torch.tensor(tokenizer.encode(t)) for t in target_texts]
    split = int(len(input_seqs) * 0.9)
    train_input, val_input = input_seqs[:split], input_seqs[split:]
    train_target, val_target = target_seqs[:split], target_seqs[split:]

    if params.get("fine_tune") and os.path.exists(selected_model_file):
        try:
            load_model(selected_model_file)
            log(f" Continuing training on existing model: {selected_model_file}")
            return  # skip re-creation to avoid mismatch
        except Exception as e:
            log(f" Failed to fine-tune existing model: {e}")
            return
    else:
        encoder = EncoderRNN(vocab_size, params['embed_size'], params['hidden_size'], dropout=0.3)
        attn = Attention(params['hidden_size'])
        decoder = DecoderRNN(vocab_size, params['embed_size'], params['hidden_size'], attn, dropout=0.3)
        model = Seq2Seq(encoder, decoder, device).to(device)
        log(" Created new model from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[PAD_TOKEN])
    scaler = GradScaler(device="cuda")

    train_losses, val_losses, val_accuracies = [], [], []
    val_bleus = []
    val_precisions, val_recalls, val_f1s = [], [], []
    val_perplexities = []

    log(f" Training started for model: {params['model_name']}")
    

    for epoch in range(params['epochs']):
        if stop_training_flag:
            save_model_checkpoint(model, params, reason="User stopped after epoch")
            return

        epoch_start_time = time.time()

        model.train()
        total_loss = 0
        combined = list(zip(train_input, train_target))
        random.shuffle(combined)
        train_input[:], train_target[:] = zip(*combined)

        for i in range(0, len(train_input), params['batch_size']):
            if stop_training_flag:
                save_model_checkpoint(model, params, reason="User stopped during batch")
                return

            inputs = torch.nn.utils.rnn.pad_sequence(
                train_input[i:i + params['batch_size']], batch_first=True,
                padding_value=tokenizer.word2idx[PAD_TOKEN]
            ).to(device)
            targets = torch.nn.utils.rnn.pad_sequence(
                train_target[i:i + params['batch_size']], batch_first=True,
                padding_value=tokenizer.word2idx[PAD_TOKEN]
            ).to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(inputs, targets)
                loss = criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train = total_loss / len(train_input)
        train_losses.append(avg_train)

        #  Validation 
        model.eval()
        val_loss = 0
        true_tokens, pred_tokens = [], []

        
        correct_tokens, total_tokens = 0, 0
        with torch.no_grad():
            for i in range(0, len(val_input), params['batch_size']):
                inputs = torch.nn.utils.rnn.pad_sequence(
                    val_input[i:i + params['batch_size']], batch_first=True,
                    padding_value=tokenizer.word2idx[PAD_TOKEN]
                ).to(device)
                targets = torch.nn.utils.rnn.pad_sequence(
                    val_target[i:i + params['batch_size']], batch_first=True,
                    padding_value=tokenizer.word2idx[PAD_TOKEN]
                ).to(device)
                log(f" Training batch {i}/{len(train_input)}... Inputs shape: {inputs.shape}")
                if torch.cuda.is_available():
                    log(f"[GPU] Used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                output = model(inputs, targets, teacher_forcing_ratio=teacher_forcing_ratio)
                pred = output.argmax(dim=2)
                # Count token-level accuracy
                mask = targets != tokenizer.word2idx[PAD_TOKEN]
                correct_tokens += (pred == targets).masked_select(mask).sum().item()
                total_tokens += mask.sum().item()


                true_ids = targets.view(-1).tolist()
                pred_ids = pred.view(-1).tolist()
                mask = [tid != tokenizer.word2idx[PAD_TOKEN] for tid in true_ids]

                true_tokens.extend([t for t, m in zip(true_ids, mask) if m])
                pred_tokens.extend([p for p, m in zip(pred_ids, mask) if m])            



                val_loss += criterion(output[:, 1:].reshape(-1, vocab_size), targets[:, 1:].reshape(-1)).item()

        avg_val = val_loss / len(val_input)
        val_losses.append(avg_val)
        val_accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
        val_accuracies.append(val_accuracy)


        precision = precision_score(true_tokens, pred_tokens, average='micro', zero_division=0)
        recall = recall_score(true_tokens, pred_tokens, average='micro', zero_division=0)
        f1 = f1_score(true_tokens, pred_tokens, average='micro', zero_division=0)

        if 'val_precisions' not in locals():
            val_precisions, val_recalls, val_f1s = [], [], []

        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        #  Compute Perplexity from Validation Loss 
        perplexity = torch.exp(torch.tensor(avg_val))
        if 'val_perplexities' not in locals():
            val_perplexities = []
        val_perplexities.append(perplexity.item())
        # Count most common true tokens (vocabulary frequency)
        token_counts = Counter(true_tokens + pred_tokens)
        top_k = top_k_tokens
        top_tokens = [token for token, _ in token_counts.most_common(top_k)]

        # Filter token predictions for top-k only
        filtered_true = [t for t, p in zip(true_tokens, pred_tokens) if t in top_tokens and p in top_tokens]
        filtered_pred = [p for t, p in zip(true_tokens, pred_tokens) if t in top_tokens and p in top_tokens]

        try:
            if len(filtered_true) > 0 and len(filtered_pred) > 0:
                cm = confusion_matrix(filtered_true, filtered_pred, labels=top_tokens)
                fig, ax = plt.subplots(figsize=(12, 10))
                token_labels = [tokenizer.idx2word.get(t, str(t)) for t in top_tokens]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=token_labels)
                disp.plot(ax=ax, xticks_rotation=45, values_format='d')
                plt.title(f"Top-{top_k} Token Confusion Matrix")
                plt.tight_layout()
                plt.savefig("static/confusion_matrix.png")
                plt.close()
                log(f"Generated confusion matrix for Top-{top_k} tokens.")
            else:
                log("Confusion Matrix skipped: not enough filtered data.")
                plt.figure()
                plt.text(0.5, 0.5, 'Confusion Matrix Skipped\n(No data)', ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.savefig("static/confusion_matrix.png")
                plt.close()

        except Exception as e:
            log(f"Confusion Matrix generation failed: {str(e)}")

        epoch_duration = time.time() - epoch_start_time
        # Now compute BLEU

        # Remove special tokens before decoding
        def filter_special_token_ids(tokens):
            return [t for t in tokens if t not in [
                tokenizer.word2idx.get("<pad>", -1),
                tokenizer.word2idx.get("<unk>", -1),
                tokenizer.word2idx.get("<start>", -1),
                tokenizer.word2idx.get("<end>", -1),
            ]]


        true_tokens = filter_special_token_ids(true_tokens)
        pred_tokens = filter_special_token_ids(pred_tokens)

        # Decode to word strings
        ref_str = tokenizer.decode(true_tokens)
        hyp_str = tokenizer.decode(pred_tokens)

        # Compute BLEU
        val_bleu = compute_bleu([ref_str], [hyp_str])

   
        log(f" Epoch {epoch+1}/{params['epochs']} | Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, Val Acc: {val_accuracy:.2f}% ,Val BLEU: {val_bleu * 100:.2f}%, {epoch_duration:.2f}s")


        # === Save Plots ===
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title("Loss Plot")
        plt.savefig("static/loss_plot.png")
        plt.close()

        plt.figure()
        plt.plot(val_accuracies, label='Validation Accuracy (%)')
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Validation Accuracy")
        plt.savefig("static/accuracy_plot.png")
        plt.close()



        val_bleus.append(val_bleu)

        plt.figure()
        plt.plot(val_bleus, label='Val BLEU')
        plt.legend()
        plt.title("BLEU Score Plot")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU")
        plt.savefig("static/bleu_plot.png")
        plt.close()

        plt.figure()
        plt.plot(val_perplexities, label='Validation Perplexity')
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.title("Perplexity over Epochs")
        plt.legend()
        plt.savefig("static/perplexity_plot.png")
        plt.close()


        plt.figure()
        plt.plot(val_precisions, label='Precision', color='blue', linestyle='--', marker='o')
        plt.plot(val_recalls, label='Recall', color='orange', linestyle='-.', marker='s')
        plt.plot(val_f1s, label='F1 Score', color='green', linestyle='-', marker='^')
        plt.title("Precision, Recall, F1 over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/prf_plot.png")
        plt.close()

        torch.cuda.empty_cache()
        

    #  Save Trained Model 
    model_path = os.path.join("models", params['model_name'])
    torch.save(model.state_dict(), model_path)

    #  Save config first
    config_path = os.path.splitext(model_path)[0] + ".json"
    with open(config_path, "w") as f:
        json.dump(params, f)

    #  Then load
    selected_model_file = model_path
    load_model(selected_model_file)
    log(f" Training complete! Model saved and loaded: {model_path}")




#  Flask Routes 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": " Empty input received."}), 400

    # Decode settings
    max_tokens = int(data.get("max_tokens", 30))
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 0))
    top_p = float(data.get("top_p", 0.0))
    decoding = data.get("decoding", "sampling")

    try:
        input_ids = tokenizer.encode(user_input.lower())
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        if decoding == "beam":
            reply = beam_search_decode(model, tokenizer, input_tensor, beam_width=3)
        elif decoding == "greedy":
            reply = generate_response(user_input, max_len=max_tokens, temperature=1.0, top_k=0, top_p=0.0)

        else:
            reply = generate_response(
                user_input,
                max_len=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        print(f"[DECODE] Method: {decoding} | Input: {user_input}")
        return jsonify({"reply": reply})
    
    except Exception as e:
        print(f"[ERROR] Chat generation failed: {e}")
        return jsonify({"reply": f" Internal error: {str(e)}"}), 500




@app.route("/train", methods=["POST"])
def trigger_train():
    global training_thread, log_messages
    log_messages.clear()
    if training_thread and training_thread.is_alive():
        return jsonify({"status": "Training already in progress."})
    params = request.json
    training_thread = threading.Thread(target=train_model, args=(params,))
    training_thread.start()
    return jsonify({"status": f"Training started for model '{params['model_name']}'"})

@app.route("/stop-training", methods=["POST"])
def stop_training():
    global stop_training_flag
    stop_training_flag = True
    return jsonify({"status": " Training manually stopped by user."})

@app.route("/train-stream")
def train_stream():
    def stream():
        previous = 0
        while True:
            time.sleep(1)
            current = len(log_messages)
            for i in range(previous, current):
                yield f"data: {log_messages[i]}\n\n"
            previous = current
    return Response(stream(), mimetype="text/event-stream")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or (not file.filename.endswith(".txt") and not file.filename.endswith(".csv")):
        return jsonify({"status": "Invalid file type. Only .txt or .csv allowed."})

    new_lines = file.read().decode("utf-8").strip().splitlines()
    dataset_path = "data/dataset.txt"

    with open(dataset_path, "a", encoding="utf-8") as f:
        for line in new_lines:
            if file.filename.endswith(".csv"):
                parts = line.split(",")
                if len(parts) >= 2:
                    f.write(f"{parts[0].strip()}|{parts[1].strip()}\n")
            else:  # .txt
                if '|' in line and line.strip():
                    f.write(line.strip() + "\n")

    return jsonify({"status": f"{len(new_lines)} lines added to dataset."})


@app.route("/plot")
def plot():
    return jsonify({
        "loss": "/static/loss_plot.png",
        "accuracy": "/static/accuracy_plot.png",
        "bleu": "/static/bleu_plot.png",
        "perplexity": "/static/perplexity_plot.png",
        "prf": "/static/prf_plot.png",
        "confusion": "/static/confusion_matrix.png",

    })

@app.route("/models")
def models():
    return jsonify([os.path.basename(f) for f in glob.glob("models/*.pt")])

@app.route("/select_model", methods=["POST"])
def select_model():
    global selected_model_file
    selected_model_file = os.path.join("models", request.json["filename"])
    load_model(selected_model_file)
    return jsonify({"status": f"Loaded model {request.json['filename']}"})

@app.route("/download/<filename>")
def download_model(filename):
    path = os.path.join("models", filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

@app.route("/dataset")
def get_dataset():
    dataset_path = "datasets2/all_input_response.txt"
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sample_lines = lines[:1000]  # Only preview first 1000 lines
        return "".join(sample_lines)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


@app.route("/clean-dataset", methods=["POST"])
def clean_dataset():

    input_path = "data/dataset.txt"
    output_path = "data/dataset.txt"
    blocklist = ["kill you", "stupid", "idiot", "dumb", "hot girls", "freak", "crazy", "slap", "punch", "hate"]

    seen = set()
    cleaned = []

    removed_too_short = 0
    removed_blocked = 0
    removed_empty = 0
    removed_duplicates = 0

    def clean_text(text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9,.?!'\" ]+", "", text)
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        try:
            corrected = str(TextBlob(text).correct())
            return corrected.strip()
        except Exception:
            return text.strip()

    def is_safe(text):
        return not any(bad in text.lower() for bad in blocklist)

    try:
        if not os.path.exists(input_path):
            return jsonify({"status": "Dataset file not found."}), 404

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" not in line:
                    continue
                input_text, target_text = map(str.strip, line.split("|", 1))

                # Word count check
                if len(input_text.split()) < 3 or len(target_text.split()) < 3:
                    removed_too_short += 1
                    continue

                if len(input_text.split()) > 30 or len(target_text.split()) > 30:
                    continue

                if not is_safe(input_text) or not is_safe(target_text):
                    removed_blocked += 1
                    continue

                input_clean = clean_text(input_text)
                target_clean = clean_text(target_text)

                if not input_clean or not target_clean:
                    removed_empty += 1
                    continue

                new_line = f"{input_clean}|{target_clean}"
                if new_line not in seen:
                    seen.add(new_line)
                    cleaned.append(new_line)
                else:
                    removed_duplicates += 1

        with open(output_path, "w", encoding="utf-8") as f:
            for line in cleaned:
                f.write(line + "\n")

        return jsonify({
            "status": f"Cleaned {len(cleaned)} lines successfully.",
            "removed_too_short": removed_too_short,
            "removed_blocked": removed_blocked,
            "removed_empty": removed_empty,
            "removed_duplicates": removed_duplicates,
            "final_dataset_size": len(cleaned)
        })

    except Exception as e:
        return jsonify({"status": "Failed to clean dataset.", "error": str(e)}), 500



@app.route("/model-config/<filename>")
def model_config(filename):
    config_path = os.path.join("models", filename.replace(".pt", ".json"))
    if not os.path.exists(config_path):
        return jsonify({})
    with open(config_path) as f:
        return jsonify(json.load(f))

def compute_bleu(references, predictions):
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for ref, pred in zip(references, predictions)
    ]
    return sum(scores) / len(scores)




def save_model_checkpoint(model, params, reason="Manual stop"):
    model_path = os.path.join("models", params["model_name"])
    torch.save(model.state_dict(), model_path)

    config_path = os.path.splitext(model_path)[0] + ".json"
    with open(config_path, "w") as f:
        json.dump(params, f)

    global selected_model_file
    selected_model_file = model_path
    load_model(selected_model_file)

    global_log(f"Model saved: {model_path} — Reason: {reason}")

def global_log(msg):
    log_messages.append(msg)
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")




if __name__ == "__main__":
    # load model on startup / just for safety messures 
    if os.path.exists(selected_model_file):
        try:
            load_model(selected_model_file)
            print(f" Auto-loaded model at startup: {selected_model_file}")
        except Exception as e:
            print(f" Failed to auto-load model: {e}")
    else:
        print(" No model file found at startup.")
    app.run(debug=True , port=5050)
