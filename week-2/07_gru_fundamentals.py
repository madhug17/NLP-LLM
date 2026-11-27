import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
sequence_lenght = 5
batch_size = 2

gru = nn.GRU(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)
x = torch.rand(batch_size,sequence_lenght,input_size)

output,hidden = gru(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)

# Update gate 
# reset gate 

