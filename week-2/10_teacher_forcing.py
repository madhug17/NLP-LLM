import torch
import torch.nn as nn

torch.manual_seed(42) # it gives same number evry time 

word_to_id = {
    "i": 0,
    "love": 1,
    "machine": 2,
    "learning": 3,
    "python": 4
}
sentence = ["i", "love", "machine", "learning"]
encoded = [word_to_id[word]for word in sentence]
# Input and target
x = torch.tensor(encoded[:-1]).unsqueeze(0)
y = torch.tensor(encoded[1:]).unsqueeze(0)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5,8)
        self.rnn = nn.RNN(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 5) # 16 is hidden layer and 5 is output layer
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)

        return logits, hidden
model = LanguageModel()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()
for epoch in range(101):

    optimizer.zero_grad()

    logits, hidden = model(x)

    loss = criterion(
        logits.reshape(-1, 5),
        y.reshape(-1)
    )
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the model
model.eval()

with torch.no_grad():
    logits, _ = model(x)
    predictions = logits.argmax(dim=-1)

print("\nActual next words:", y.tolist())
print("Predicted IDs:", predictions.tolist())
