import torch
import torch.nn as nn
input_size = 5
hidden_size = 8
rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)
x = torch.randn(2,4,5)
# 1 is batch size and 4 is Sequence lenght and 5 is Features per step
# example : machine learning is powerful
output, hidden = rnn(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)