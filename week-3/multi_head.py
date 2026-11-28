import torch
import torch.nn as nn

embedding_dim = 8
num_heads = 2

multi_attention = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=num_heads,
    batch_first=True
)

# Batch size = 1, tokens = 3, embedding = 8
X = torch.randn(1, 3, 8)

output, attention_weights = multi_attention(
    X, X, X
)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights:", attention_weights.shape)
print(attention_weights)