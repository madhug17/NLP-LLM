# Input = TokenEmbedding + PositionalEncoding


import torch
import math

sequence_length = 4
embedding_dim = 8

position = torch.arange(sequence_length).unsqueeze(1)

dimensions = torch.arange(0, embedding_dim, 2)

div_term = torch.exp(
    dimensions * (-math.log(10000.0) / embedding_dim)
)

PE = torch.zeros(sequence_length, embedding_dim)

PE[:, 0::2] = torch.sin(position * div_term)
PE[:, 1::2] = torch.cos(position * div_term)

print("Positional Encoding:")
print(PE)

print("Shape:", PE.shape)
