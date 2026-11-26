import torch
x = torch.tensor([[0.5,0.8]])
h_prev = torch.tensor([[0.2, 0.4]])
c_prev = torch.tensor([[0.3, 0.6]])
W_f = torch.tensor([[0.2, 0.1],
                    [0.4, 0.3]])
U_f = torch.tensor([[0.1, 0.2],
                    [0.2, 0.1]])
b_f = torch.zeros(1, 2)
W_i = W_f.clone()
U_i = U_f.clone()
b_i = torch.zeros(1, 2)

W_c = W_f.clone()
U_c = U_f.clone()
b_c = torch.zeros(1, 2)
W_o = W_f.clone()
U_o = U_f.clone()
b_o = torch.zeros(1, 2)

sigmoid = torch.sigmoid
tanh = torch.tanh

# Forget gate
f = sigmoid(x @ W_f.T + h_prev @ U_f.T + b_f)
# Input gate
i = sigmoid(x @ W_i.T + h_prev @ U_i.T + b_i)
# Candidate cell state
c_candidate = tanh(x @ W_c.T + h_prev @ U_c.T + b_c)
# Output gate
o = sigmoid(x @ W_o.T + h_prev @ U_o.T + b_o)
# Update cell state
c = f * c_prev + i * c_candidate
# Update hidden state
h = o * tanh(c)

print("Forget gate:", f)
print("Input gate:", i)
print("Candidate:", c_candidate)
print("Output gate:", o)
print("New cell state:", c)
print("New hidden state:", h)