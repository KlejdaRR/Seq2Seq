import torch
import spacy
from models.encoder import Encoder as Encoder
from models.decoder import Decoder as Decoder
from models.seq2seq import Seq2Seq
from utils.dataset import Vocabulary
from utils.evaluate import translate_sentence
from utils.train import load_checkpoint
from utils.dataset import Multi30kDataset
from sklearn.model_selection import train_test_split

# Load spacy tokenizer for Italian
spacy_it = spacy.load("it_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Multi30kDataset('data/train.en.txt', 'data/train.it.txt')
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

def tokenize_it(text):
    return [tok.text.lower() for tok in spacy_it.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

torch.serialization.add_safe_globals([Vocabulary])

# Load vocabularies
italian_vocab = torch.load("italian_vocab.pth", weights_only=False)
english_vocab = torch.load("english_vocab.pth", weights_only=False)

# Load model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_net = Encoder(len(italian_vocab.word2index), 300, 1024, 2, 0.5).to(device)
decoder_net = Decoder(len(english_vocab.word2index), 300, 1024, len(english_vocab.word2index), 2, 0.5).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)

# Load trained model checkpoint
checkpoint = torch.load("my_checkpoint.pth", map_location=device)
load_checkpoint(checkpoint, model, optimizer=None)  # No need for optimizer in inference mode
model.eval()  # Set model to evaluation mode

def translate_user_input(model, italian_vocab, english_vocab, device):
    while True:
        sentence = input("\nEnter an Italian sentence (or type 'exit' to quit): ")
        if sentence.lower() == "exit":
            break
        translated_sentence = translate_sentence(model, sentence, italian_vocab, english_vocab, device)
        print("Translated Sentence:", " ".join(translated_sentence))

# Load the saved model before using it for translation
checkpoint = torch.load("my_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()  # Set model to evaluation mode

# Start interactive translation
translate_user_input(model, italian_vocab, english_vocab, device)

