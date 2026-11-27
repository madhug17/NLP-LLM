
import torch
import torch.nn as nn

torch.manual_seed(42)

word_to_id = {
    "i": 0,
    "love": 1,
    "machine": 2,
    "learning": 3,
    "python": 4,
    "<EOS>": 5
}

id_to_word = {
    index: word
    for word, index in word_to_id.items()
}

sentence = ["i", "love", "machine", "learning"]

encoded = [word_to_id[word] for word in sentence]

x = torch.tensor([encoded[:-1]])
y = torch.tensor([encoded[1:]])

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(6, 8)
        self.gru = nn.GRU(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 6)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output)

        return logits, hidden


model = LanguageModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(301):

    optimizer.zero_grad()

    logits, _ = model(x)

    loss = criterion(
        logits.reshape(-1, 6),
        y.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Autoregressive generation
model.eval()

current_input = torch.tensor([[word_to_id["i"]]])
hidden = None

generated = ["i"]

with torch.no_grad():

    for _ in range(5):

        logits, hidden = model(current_input, hidden)

        predicted_id = logits[:, -1, :].argmax(dim=-1)

        word = id_to_word[predicted_id.item()]

        if word == "<EOS>":
            break

        generated.append(word)

        current_input = predicted_id.unsqueeze(0)

print("\nGenerated sentence:", " ".join(generated))
