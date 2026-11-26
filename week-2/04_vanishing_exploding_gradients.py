import torch
import torch.nn as nn
input_size = 10
hidden_size = 20
rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)
x = torch.rand(1,100,10)
output , hidden = rnn(x)
loss = output.sum()
loss.backward()
for seq_length in [10, 50, 100, 200, 500, 1000]:

    rnn.zero_grad()

    x = torch.randn(1, seq_length, 10)

    output, hidden = rnn(x)

    loss = output.sum()

    loss.backward()

    print(
        f"Sequence: {seq_length}, "
        f"Gradient: {rnn.weight_hh_l0.grad.norm().item():.6f}"
    )