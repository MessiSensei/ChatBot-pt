
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib  as plt
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf') if attention.dtype == torch.float32 else -1e4)

        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, attention, dropout=0.1, pad_token_idx=0):
        super(DecoderRNN, self).__init__()
        self.pad_token_idx = pad_token_idx
        self.output_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input, hidden, encoder_outputs, src=None, return_attention=False):
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input))  # (batch, 1, embed_size)

        # Apply mask using stored pad_token_idx
        src_mask = (src != self.pad_token_idx).int() if src is not None else None

        # Compute attention weights with optional mask
        attn_weights = self.attention(hidden, encoder_outputs, mask=src_mask).unsqueeze(1)

        # Compute context vector
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden)
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed+hidden)

        # GRU output
        output, hidden = self.gru(rnn_input, hidden)  # output: (batch, 1, hidden)

        # Final prediction
        prediction = self.fc(torch.cat((output, context), dim=2)).squeeze(1)  # (batch, vocab_size)

        if return_attention:
            return prediction, hidden, attn_weights
        else:
            return prediction, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, output_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0] 

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, src)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs



def beam_search_decode(model, tokenizer, input_tensor, beam_width=3, max_len=30):
    device = input_tensor.device
    START = tokenizer.word2idx.get("<start>", 2)
    END = tokenizer.word2idx.get("<end>", 3)
    PAD = tokenizer.word2idx.get("<pad>", 0)

    encoder_outputs, hidden = model.encoder(input_tensor)
    
    # Initialize beam: (sequence, hidden_state, log_prob, attention_list)
    beams = [(torch.tensor([START], device=device), hidden, 0.0, [])]

    for _ in range(max_len):
        all_candidates = []
        for seq, h, log_prob, attns in beams:
            input_token = seq[-1].unsqueeze(0)
            output, h_new, attn = model.decoder(input_token, h, encoder_outputs, return_attention=True)

            probs = torch.softmax(output.squeeze(0), dim=-1)
            topk_probs, topk_indices = torch.topk(probs, beam_width)

            for i in range(beam_width):
                token = topk_indices[i].item()
                token_log_prob = torch.log(topk_probs[i] + 1e-10).item()
                new_seq = torch.cat([seq, torch.tensor([token], device=device)])
                new_attns = attns + [attn]
                all_candidates.append((new_seq, h_new, log_prob + token_log_prob, new_attns))

        # Sort by score and retain top beams
        beams = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:beam_width]

        # Early stopping if all beams ended
        if all(seq[-1].item() == END for seq, _, _, _ in beams):
            break

    # Select the best beam
    best_seq, _, _, _ = beams[0]

    # Remove special tokens
    filtered_seq = [
        t.item() for t in best_seq 
        if t.item() not in {START, END, PAD}
    ]

    return tokenizer.decode(filtered_seq)



#  Attention Heatmap Visualization
def get_attention_heatmap(model, tokenizer, input_text):
    model.eval()
    tokens = tokenizer.encode(input_text.lower())
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(model.device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(input_tensor)
        input_token = torch.tensor([tokenizer.word2idx["<start>"]], device=model.device)
        attention_weights = []
        decoded_tokens = []

        for _ in range(20):
            output, hidden, attn = model.decoder(input_token, hidden, encoder_outputs, return_attention=True)
            top1 = output.argmax(1).item()
            input_token = torch.tensor([top1], device=model.device)
            if top1 == tokenizer.word2idx["<end>"]:
                break
            decoded_tokens.append(top1)
            attention_weights.append(attn.squeeze(1).cpu().numpy())

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            attention_weights,
            xticklabels=[tokenizer.idx2word[i] for i in tokens],
            yticklabels=[tokenizer.idx2word[i] for i in decoded_tokens],
            cmap="viridis",
            ax=ax
        )
        ax.set_xlabel("Input Sequence")
        ax.set_ylabel("Generated Output")
        plt.tight_layout()
        plt.savefig("static/attention_heatmap.png")
        plt.close()