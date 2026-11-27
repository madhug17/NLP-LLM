import torch
x = torch.tensor([[0.5, 0.8]])
h_prev = torch.tensor([[0.2, 0.4]])

# Weights
W_z = torch.tensor([
    [0.2, 0.1],
    [0.4, 0.3]
])

U_z = torch.tensor([
    [0.1, 0.2],
    [0.2, 0.1]
])

W_r = W_z.clone()
U_r = U_z.clone()
W_h = W_z.clone()
U_h = U_z.clone()
b_z = torch.zeros(1, 2)
b_r = torch.zeros(1, 2)
b_h = torch.zeros(1, 2)

# Update gate
z = torch.sigmoid(
    x @ W_z.T + h_prev @ U_z.T + b_z
)

# Reset gate
r = torch.sigmoid(
    x @ W_r.T + h_prev @ U_r.T + b_r
)

# Candidate hidden state
h_candidate = torch.tanh(
    x @ W_h.T + (r * h_prev) @ U_h.T + b_h
)

# New hidden state
h = (1 - z) * h_candidate + z * h_prev

print("Update gate:", z)
print("Reset gate:", r)
print("Candidate hidden:", h_candidate)
print("New hidden state:", h)