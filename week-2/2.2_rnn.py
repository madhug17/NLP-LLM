import torch
import torch.nn as nn
input_size = 5
hidden_size = 8
rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)
x = torch.rand(1,4,5)
output,hidden = rnn(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)
print("\nOutput for first word:", output[:, 0, :])
print("Output for second word:", output[:, 1, :])
print("Output for third word:", output[:, 2, :])
print("Output for fourth word:", output[:, 3, :])
print("\nFinal hidden state:", hidden)