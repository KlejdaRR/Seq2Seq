import torch
import spacy
from models.encoder import Encoder as Encoder
from models.Seq2SeqWithAttention import Seq2SeqWithAttention
from models.DecoderWithAttention import DecoderWithAttention
from utils.dataset import Vocabulary
from utils.evaluate import translate_sentence
from utils.train import load_checkpoint
spacy_it = spacy.load("it_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_it(text):
    return [tok.text.lower() for tok in spacy_it.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

torch.serialization.add_safe_globals([Vocabulary])

# Loading vocabularies
italian_vocab = torch.load("italian_vocab.pth", weights_only=False)
english_vocab = torch.load("english_vocab.pth", weights_only=False)

# Loading model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_net = Encoder(len(italian_vocab.word2index), 300, 1024, 2, 0.5).to(device)
decoder_net = DecoderWithAttention(len(english_vocab.word2index), 300, 1024, len(english_vocab.word2index), 2, 0.5).to(device)
model = Seq2SeqWithAttention(encoder_net, decoder_net).to(device)

# Loading trained model checkpoint
checkpoint = torch.load("my_checkpoint.pth", map_location=device)
load_checkpoint(checkpoint, model, optimizer=None)
model.eval()  # Set model to evaluation mode

def translate_user_input(model, italian_vocab, english_vocab, device):
    while True:
        sentence = input("\nEnter an Italian sentence (or type 'exit' to quit): ")
        if sentence.lower() == "exit":
            break
        translated_sentence = translate_sentence(model, sentence, italian_vocab, english_vocab, device)
        print("Translated Sentence:", " ".join(translated_sentence))

# Starting interactive translation
translate_user_input(model, italian_vocab, english_vocab, device)

