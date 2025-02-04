import torch
import torch.nn as nn
from models.LuongAttention import LuongAttention

class DecoderWithAttention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = LuongAttention(hidden_size)  # Luong Attention mechanism

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedding = self.dropout(self.embedding(x))
        context, _ = self.attention(hidden[-1].unsqueeze(0), encoder_outputs)
        rnn_input = torch.cat((embedding, context.unsqueeze(1)), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))

        return self.softmax(predictions), hidden, cell

