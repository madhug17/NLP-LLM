# in this it will create it self Q,K,V values from nn.Linear

import torch
import torch.nn as nn
import torch.nn.functional as F # Function is used for the using stateless function less sigmoid , relu, softmax
import math

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0]
])

embedding_dim = 4

W_query = nn.Linear(embedding_dim, 4)
W_key = nn.Linear(embedding_dim, 4)
W_value = nn.Linear(embedding_dim, 4)

Q = W_query(X)
K = W_key(X)
V = W_value(X)

scores = Q @ K.T

d_k = K.shape[-1]
scaled_scores = scores / math.sqrt(d_k)

weights = F.softmax(scaled_scores, dim=-1)

context = weights @ V

print("Input:", X.shape)
print("Query:", Q.shape)
print("Key:", K.shape)
print("Value:", V.shape)
print("Scores:", scores.shape)
print("Weights:", weights)
print("Context:", context)
print("Context shape:", context.shape)