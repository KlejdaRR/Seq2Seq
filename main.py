from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.LuongAttention import LuongAttention
from models.Seq2SeqWithAttention import Seq2SeqWithAttention
from models.DecoderWithAttention import DecoderWithAttention
from utils.dataset import Multi30kDataset, tokenize_it, tokenize_eng, Vocabulary, collate_fn
from utils.train import train_model, save_checkpoint, evaluate_model
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load dataset
dataset = Multi30kDataset('data/train.en.txt', 'data/train.it.txt')
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Build vocabulary
italian_vocab = Vocabulary()
english_vocab = Vocabulary()

for src, tgt in train_data:
    italian_vocab.add_sentence(tokenize_it(src))
    english_vocab.add_sentence(tokenize_eng(tgt))

# Print vocabulary sizes
print("Italian Vocabulary Size:", len(italian_vocab))
print("English Vocabulary Size:", len(english_vocab))

# Print some words from the vocabulary
print("Sample Italian Words:", list(italian_vocab.word2index.keys())[:10])
print("Sample English Words:", list(english_vocab.word2index.keys())[:10])

# Create data loaders
train_iterator = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=lambda x: collate_fn(x, italian_vocab, english_vocab), drop_last=True)
valid_iterator = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, italian_vocab, english_vocab), drop_last=True)
test_iterator = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, italian_vocab, english_vocab), drop_last=True)

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_net = Encoder(len(italian_vocab.word2index), 300, 1024, 2, 0.5).to(device)
decoder_net = DecoderWithAttention(len(english_vocab.word2index), 300, 1024, len(english_vocab.word2index), 2, 0.5).to(device)
model = Seq2SeqWithAttention(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Training loop
best_valid_loss = float('inf')
for epoch in range(100):
    print(f"[Epoch {epoch + 1}/100]")
    train_loss = train_model(model, train_iterator, optimizer, criterion, device)
    valid_loss = evaluate_model(model, valid_iterator, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="my_checkpoint.pth")

# Save vocabularies for consistency during inference
torch.save(italian_vocab, "italian_vocab.pth")
torch.save(english_vocab, "english_vocab.pth")