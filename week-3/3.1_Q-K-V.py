import torch
import torch.nn.functional as F

query = torch.tensor([[1.0, 0.0]])
keys =  torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.8, 0.2]
])
values = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [8.0, 2.0]
])
'''
Query: one vector representing what we're looking for.
Keys: three vectors to compare against the Query.
Values: three vectors containing the information we can retrieve.

'''

# Score is for to find the difference between Query and Key 
scores = query@keys.T # @ is will helps of matrix multiplication
weights = F.softmax(scores,dim=1) 
print("Attention Scores:")
print(scores)
print("Attention Weights:")
print(weights) # Determine how much to focus on each token

context = weights @ values #This combines the Values according to their attention weights.
print("Context Vector:")
print(context)

'''
Values = information available from each word.
Weights = how much importance to give each word.
Context Vector = the combined information.
'''

import math 
query = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

keys = torch.tensor([
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0]
])

values = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0],
    [50.0, 60.0]
])

# Step 1: Calculate dot product
scores = query @ keys.T
print("Original scores:", scores)

# Step 2: Calculate scaling factor
d_k = keys.shape[-1]
scale = math.sqrt(d_k)

# Step 3: Scale scores
scaled_scores = scores / scale
print("Scaled scores:", scaled_scores)

# Step 4: Apply Softmax
weights = F.softmax(scaled_scores, dim=-1)
print("Attention weights:", weights)

# Step 5: Calculate context vector
context = weights @ values
print("Context vector:", context)
