import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu

# Load SpaCy tokenizers for Italian and English
spacy_it = spacy.load("it_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_it(text):
    return [tok.text.lower() for tok in spacy_it.tokenizer(text)]

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
def bleu(data, model, german_vocab, english_vocab, device):
    targets = []
    outputs = []

    for example in data:
        src = example[0]
        trg = example[1]

        prediction = translate_sentence(model, src, german_vocab, english_vocab, device)
        prediction = prediction[:-1]

        targets.append([trg.split()])
        outputs.append(prediction)

    return sum(sentence_bleu(ref, pred) for ref, pred in zip(targets, outputs)) / len(outputs)