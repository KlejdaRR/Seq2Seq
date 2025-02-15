import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import torch.nn.functional as F


# Load SpaCy tokenizers for Italian and English
spacy_it = spacy.load("it_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_it(text):
    return [tok.text.lower() for tok in spacy_it.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

# Function to translate a sentence from Italian to English using a trained model
def translate_sentence(model, sentence, italian_vocab, english_vocab, device, max_length=50):
    tokens = tokenize_it(sentence)
    tokens = ["<sos>"] + tokens + ["<eos>"]
    input_indices = [italian_vocab.word2index.get(token, 3) for token in tokens]  # 3 is <unk>

    sentence_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell, encoder_outputs = model.encoder(sentence_tensor)

    outputs = [english_vocab.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell, encoder_outputs)

        best_guess = output.argmax(1).item()
        outputs.append(best_guess)

        if best_guess == english_vocab.word2index["<eos>"]:
            break

    translated_sentence = [english_vocab.index2word.get(idx, "<unk>") for idx in outputs]

    return translated_sentence[1:-1]

# Function to compute the BLEU score for evaluating translation quality
def bleu(data, model, italian_vocab, english_vocab, device):
    chencherry = SmoothingFunction()  # Smoothing function for BLEU
    targets = []
    outputs = []

    for example in data:
        src = example[0]
        trg = example[1]

        prediction = translate_sentence(model, src, italian_vocab, english_vocab, device)
        prediction = prediction[:-1]  # Remove final token

        targets.append([trg.split()])
        outputs.append(prediction)

    return sum(sentence_bleu(ref, pred, smoothing_function=chencherry.method1) for ref, pred in zip(targets, outputs)) / len(outputs)



def calculate_perplexity(model, data, italian_vocab, english_vocab, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for example in data:
            src = example[0]
            trg = example[1]

            src_tokens = ["<sos>"] + tokenize_it(src) + ["<eos>"]
            src_indices = [italian_vocab.word2index.get(token, 3) for token in src_tokens]
            src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

            trg_tokens = ["<sos>"] + tokenize_eng(trg) + ["<eos>"]
            trg_indices = [english_vocab.word2index.get(token, 3) for token in trg_tokens]
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

            output = model(src_tensor, trg_tensor, teacher_force_ratio=0)

            output = output[:, 1:].reshape(-1, output.shape[2])
            trg = trg_tensor[:, 1:].reshape(-1)

            loss = F.cross_entropy(output, trg, ignore_index=english_vocab.word2index["<pad>"])
            total_loss += loss.item() * (trg != english_vocab.word2index["<pad>"]).sum().item()
            total_tokens += (trg != english_vocab.word2index["<pad>"]).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity