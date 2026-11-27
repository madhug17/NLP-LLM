import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10,8) # 10 is number of words and 8 is number of numerical values
        self.gru = nn.GRU(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )
    def forward(self,x):
        embedded = self.embedding(x)
        output,hidden= self.gru(embedded)
        return hidden
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10,8)
        self.gru = nn.GRU(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )
        self.fc = nn.Linear(16,10)
    def forward(self,x,hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded,hidden)
        predictions = self.fc(output)
        return predictions,hidden
encode = Encoder()
decode = Decoder()
source = torch.tensor([[1,2,3,4,5]])
encode_hidden = encode(source)
# Decoder input (starting token)
decoder_input = torch.tensor([[0]])
output, decoder_hidden = decode(
    decoder_input,
    encode_hidden
)
print("Source shape:", source.shape)
print("Encoder hidden:", encode_hidden.shape)
print("Decoder input:", decoder_input.shape)
print("Decoder output:", output.shape)
print("Decoder hidden:", decoder_hidden.shape)