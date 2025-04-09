import torch
from model import EncoderRNN, Attention, DecoderRNN, Seq2Seq
from tokenizer import SimpleTokenizer
import glob
import os

#  Config 
EMBED_SIZE = 32
HIDDEN_SIZE = 64
VOCAB_FILE = "vocab.txt"
MAX_LEN = 30

# Special tokens
START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"

#  Load models 
model_files = glob.glob("models/*.pt")
if not model_files:
    raise FileNotFoundError("No model files found.")

print("Available models:")
for i, f in enumerate(model_files):
    print(f"[{i}] {os.path.basename(f)}")

choice = input("Select a model to load by number (or press Enter for latest): ").strip()
MODEL_FILE = max(model_files, key=os.path.getctime) if choice == "" else model_files[int(choice)]
print(f"Loading model: {MODEL_FILE}")

#  Load tokenizer 
tokenizer = SimpleTokenizer()
tokenizer.load_vocab(VOCAB_FILE)
vocab_size = len(tokenizer.word2idx)

start_id = tokenizer.word2idx[START_TOKEN]
end_id = tokenizer.word2idx[END_TOKEN]
pad_id = tokenizer.word2idx[PAD_TOKEN]

#  Build model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
attn = Attention(HIDDEN_SIZE)
decoder = DecoderRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, attn, pad_token_idx=pad_id)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

#  Generate Response 
def generate_response(input_text):
    input_tokens = tokenizer.encode(input_text.lower())
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    encoder_outputs, hidden = model.encoder(input_tensor)
    input_token = torch.tensor([start_id], dtype=torch.long).to(device)

    response_ids = []
    for _ in range(MAX_LEN):
        output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs, src=input_tensor, return_attention=True)
        top1 = output.argmax(1).item()
        if top1 in [end_id, pad_id]:
            break
        response_ids.append(top1)
        input_token = torch.tensor([top1], dtype=torch.long).to(device)

    return tokenizer.decode(response_ids)

#  CLI Chat Loop 
if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        reply = generate_response(inp)
        print("Bot:", reply)
