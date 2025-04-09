import re
from collections import Counter
from nltk.tokenize import word_tokenize

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def _tokenize(self, text):
        return word_tokenize(text.lower())

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        vocab = ["<pad>", "<unk>", "<start>", "<end>"] + sorted(counter.keys())
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text):
        tokens = ["<start>"] + self._tokenize(text) + ["<end>"]
        unk_idx = self.word2idx.get("<unk>", 1)
        return [self.word2idx.get(word, unk_idx) for word in tokens]

    def decode(self, indices):
        tokens = [self.idx2word.get(idx, "<unk>") for idx in indices]
        sentence = " ".join(tokens)
        sentence = re.sub(r"\s+([?.!,\"'])", r"\1", sentence)  
        sentence = sentence.replace("<pad>", "").replace("<unk>", "").strip()
        return sentence


    def save_vocab(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")

    def load_vocab(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word, idx = line.strip().split("\t")
                self.word2idx[word] = int(idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
