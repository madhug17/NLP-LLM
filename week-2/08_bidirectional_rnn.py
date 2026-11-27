import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2

rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True,
    bidirectional=True
)

x = torch.rand(batch_size,sequence_length,input_size)
output, hidden = rnn(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape) ## output is 40 because of 20 + 20 left to right and right to left 
print("Hidden shape:", hidden.shape)