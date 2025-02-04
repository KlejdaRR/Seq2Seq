import torch
import spacy
from nltk.translate.bleu_score import sentence_bleu

def translate_sentence(model, sentence, italian_vocab, english_vocab, device, max_length=50):
    spacy_it = spacy.load("it_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_it(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    text_to_indices = [italian_vocab.word2index.get(token, 3) for token in tokens]  # 3 is <unk>

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english_vocab.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if best_guess == english_vocab.word2index["<eos>"]:
            break

    translated_sentence = [english_vocab.index2word[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]

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