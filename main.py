from models.encoder import Encoder
from models.Seq2SeqWithAttention import Seq2SeqWithAttention
from models.DecoderWithAttention import DecoderWithAttention
from utils.dataset import Multi30kDataset, tokenize_it, tokenize_eng, Vocabulary, collate_fn
from utils.train import train_model, save_checkpoint
from utils.evaluate import bleu, calculate_perplexity
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load dataset (Italian -> English)
dataset = Multi30kDataset('data/train.it.txt', 'data/train.en.txt')
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Build vocabulary
italian_vocab = Vocabulary()
english_vocab = Vocabulary()

for src, tgt in train_data:
    italian_vocab.add_sentence(tokenize_it(src))
    english_vocab.add_sentence(tokenize_eng(tgt))

italian_vocab.finalize_vocab()
english_vocab.finalize_vocab()

torch.save(italian_vocab, "italian_vocab.pth")
torch.save(english_vocab, "english_vocab.pth")

# Creation data loaders
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

for epoch in range(100):
    print(f"[Epoch {epoch + 1}/100]")
    loss = train_model(model, train_iterator, optimizer, criterion, device)
    print(f"Train Loss: {loss:.4f}")

    if epoch % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.7

    if epoch % 10 == 0:
        with torch.no_grad():
            sample_src, sample_tgt = next(iter(train_iterator))
            sample_src = sample_src[0].unsqueeze(0).to(device)
            sample_tgt = sample_tgt[0].unsqueeze(0).to(device)
            output = model(sample_src, sample_tgt, teacher_force_ratio=0)  # No teacher forcing in inference

        output_indices = output.argmax(2).squeeze().tolist()
        translated_sentence = [english_vocab.index2word.get(idx, "<unk>") for idx in output_indices]

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename="my_checkpoint.pth")

bleu_score = bleu(test_data, model, italian_vocab, english_vocab, device)
print(f"\nFinal BLEU Score on Test Data: {bleu_score:.4f}")

perplexity_score = calculate_perplexity(model, test_data, italian_vocab, english_vocab, device)
print(f"Perplexity Score: {perplexity_score}")