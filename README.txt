## Seq2Seq Machine Translator
This project implements a Sequence-to-Sequence (Seq2Seq) model for automatic translation using PyTorch.
The model translates sentences from Italian to English using an encoder-decoder architecture with Luong Attention LSTM layers.

## Features
Uses two dataset files (train.en.txt, train.it.txt) with Italian-English sentence pairs.
Implements a Seq2Seq model with an attention mechanism.
For better accuracy and model convergence, implements Luong Attention Mechanism.
Supports training and inference using PyTorch.
Tokenization with spaCy.
Data loading and batching with PyTorch Dataset.
The dataset includes two text files (train.en.txt, train.it.txt) with sentence pairs.

## Installation prerequisites
python: 3.9
torch: 2.2.2
torchtext: 0.17.2
spacy: 3.8.4
numpy: 1.26.4
spaCy language models:
python -m spacy download it_core_news_sm python -m spacy download en_core_web_sm

## Training the Model
Run: python main.py

## Running Inference
Testing of translations is done by using: python run.py

Enter an Italian sentence, and the model will generate an English translation.

## Project Structure
Seq2SeqMachineTranslator/
│── data/
│ ├── train.en.txt # English sentences
│ ├── train.it.txt # Italian sentences
│── models/
│ ├── encoder.py # Encoder model
│ ├── decoder.py # Decoder model
│ ├── seq2seq.py # Seq2Seq architecture
│ ├── DecoderWithAttention.py # decoder class updated with Luong Attention Mechanism
│ ├── LuongAttention.py
│ ├── Seq2SeqWithAttention.py # Seq2Seq class updated with Luong Attention Mechanism
│── utils/
│ │── dataset.py # Data loading & tokenization
│ ├── train.py # Training functions
│ ├── evaluate.py # Inference functions
│── main.py # Training script
│── run.py # Translation script
│── README.md # Project documentation

## Project's contributors:
Klejda Rrapaj: k.rrapaj@student.unisi.it
Sildi Ricku: s.ricku@student.unisi.it