sentences = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "python is a programming language",
    "artificial intelligence is growing",
    "machine learning uses algorithms"
]
tokenization = [
    sentence.lower().split()
    for sentence in sentences
]
vocal = sorted(set(
    word
    for sentence in tokenization
    for word in sentence
))
word_to_id = {
    word:index
    for index,word in enumerate(vocal)
}
#print("Vocabulary:", vocal)
#print("Word to ID:", word_to_id)


encoded_sentences = [
    [word_to_id[word] for word in sentence]
    for sentence in tokenization
]
print(encoded_sentences)

## next we are going to predict the words 
training_data= []
window_size =2
for sentence in encoded_sentences:
    for i in range(len(sentence)-window_size):
        context = sentence[i:i+2]
        target = sentence[i+2]
        training_data.append((context,target))
print(training_data)

#converting this in to pytorch

import torch

X = [context for context, target in training_data]
y = [target for context, target in training_data]
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
print("X:", X)
print("y:", y)
print("X shape:", X.shape)
print("y shape:", y.shape)

## Using the nn.embedding , nn.Linear,nn.ReLu
import torch
import torch.nn as nn
vocab_size = len(word_to_id)
embedding_dim = 16
hidden_dim = 32
class FixedWindowLM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.reLu = nn.ReLU()
        self.fc2 = nn.Linear (hidden_dim, vocab_size)
    def forward(self,x):
        embeddings = self.embedding(x)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        hidden = self.sigmoid(self.fc1(embeddings))
        output = self.fc2(hidden)
        return output
model = FixedWindowLM(
    vocab_size,
    embedding_dim,
    hidden_dim
)

print(model)

predictions = model(X)
print("Predictions:", predictions)
print("Predictions shape:", predictions.shape)

## CrossEntropyLoss

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)
for epoch in range(100):
    optimizer.zero_grad()  # Clear old gradients
    predictions = model(X) # Forward propagation
    loss = criterion(predictions, y)
    loss.backward()        # Calculate new gradients
    optimizer.step()       # Update weights
    if (epoch+1) % 10==0:
        print(f"epoch {epoch+1}, Loss: {loss.item():.4f}")
model.eval()
with torch.no_grad():
    test_input = torch.tensor(
        [[word_to_id['deep'],word_to_id['learning']]],
        dtype=torch.long
    )
    output = model(test_input)
    predicted_id = torch.argmax(output, dim=1).item()
    predicted_word = next(
        word for word, idx in word_to_id.items()
        if idx == predicted_id
    )
    
    print("Input:", "deep learning")
    print("Predicted next word:", predicted_word)
