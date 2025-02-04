import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu

def translate_sentence(model, sentence, italian_vocab, english_vocab, device, max_length=50):
    spacy_it = spacy.load("it_core_news_sm")

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_it(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Debug: Print input tokens
    print("Input tokens:", tokens)

    tokens = ["<sos>"] + tokens + ["<eos>"]
    text_to_indices = [italian_vocab.word2index.get(token, 3) for token in tokens]  # 3 is <unk>

    # Debug: Print input indices
    print("Input indices:", text_to_indices)

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

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

    # Debug: Print output indices
    print("Output indices:", outputs)

    translated_sentence = [english_vocab.index2word[idx] for idx in outputs]

    # Debug: Print translated sentence
    print("Translated sentence:", translated_sentence)

    return translated_sentence[1:]  # Exclude <sos>

def bleu(data, model, italian_vocab, english_vocab, device):
    targets = []
    outputs = []

    for example in data:
        src = example[0]  # Source sentence
        trg = example[1]  # Target sentence

        prediction = translate_sentence(model, src, italian_vocab, english_vocab, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg.split()])
        outputs.append(prediction)

    # Compute BLEU score manually using nltk
    return sum(sentence_bleu(ref, pred) for ref, pred in zip(targets, outputs)) / len(outputs)