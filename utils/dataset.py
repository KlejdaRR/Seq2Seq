import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# Load spacy tokenizers
spacy_it = spacy.load("it_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

class Multi30kDataset(Dataset):
    def __init__(self, source_file, target_file):
        with open(source_file, 'r', encoding='utf-8') as f:
            self.source_sentences = f.readlines()
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_sentences = f.readlines()

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        return self.source_sentences[idx].strip(), self.target_sentences[idx].strip()

def tokenize_it(text):
    return [tok.text.lower() for tok in spacy_it.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

class Vocabulary:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_count = {}

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

def collate_fn(batch, italian_vocab, english_vocab):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_tokens = [italian_vocab.word2index.get(word, 3) for word in tokenize_it(src)] + [2]  # <eos>
        tgt_tokens = [english_vocab.word2index.get(word, 3) for word in tokenize_eng(tgt)] + [2]  # <eos>
        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_tokens, dtype=torch.long))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch