import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_length = 5
batch_size = 2
lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)
# Generate sample input
x = torch.rand(batch_size, sequence_length, input_size)

# Forward Pass
output,(hidden,cell) = lstm(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Hidden state shape:", hidden.shape)
print("Cell state shape:", cell.shape)
