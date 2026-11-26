import torch
import torch.nn as nn
sentence = "machine learning is powerful"
words = sentence.split()
vocab = sorted(set(words))
word_to_id = {
    word:index
    for index,word in enumerate(vocab)
}
id_to_word = {
    index:word
    for word,index in word_to_id.items()
}
encode = [word_to_id[word] for word in words] 
x = torch.tensor([encode[:-1]],dtype=torch.long)
y = torch.tensor([encode[1:]],dtype=torch.long)

## RNN FORWARD AND BACKWARD TRAINING
class RNNLanguage(nn.Module):
    def __init__ (self,vocal_size,embedding_dim,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocal_size,embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,vocal_size)
    def forward(self,x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        predictions = self.fc(output)
        return predictions
 ## creating model
model = RNNLanguage(
     vocal_size=len(vocab),
     embedding_dim=5,
     hidden_size=8
 )
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(
        predictions.reshape(-1, len(vocab)),
        y.reshape(-1)
    )
    loss.backward()
    optimizer.step()
    if (epoch+1)%20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# model prediction
model.eval()
with torch.no_grad():
    test_input = torch.tensor(
        [[word_to_id["machine"], word_to_id["learning"]]],
        dtype=torch.long
    )
    predictions = model(test_input)
    next_word_id = predictions[:, -1, :].argmax(dim=1).item()
    print("\nVocabulary:", word_to_id)
    print("Reverse vocabulary:", id_to_word)
    print("Predicted ID:", next_word_id)
    print("Predicted next word:", id_to_word.get(next_word_id, "Unknown"))








