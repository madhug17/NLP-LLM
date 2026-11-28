
import torch
import torch.nn as nn
import string

# ==================================
# 1. LOAD DATASET
# ==================================

with open("ai_ml_corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))

# ==================================
# 2. TEXT PREPROCESSING
# ==================================

clean_text = text.lower().translate(
    str.maketrans("", "", string.punctuation)
)

words = clean_text.split()

print("Total words:", len(words))
print("First 30 words:", words[:30])

# ==================================
# 3. BUILD VOCABULARY
# ==================================

vocab = sorted(set(words))

word_to_id = {
    word: index
    for index, word in enumerate(vocab)
}

id_to_word = {
    index: word
    for word, index in word_to_id.items()
}

print("Vocabulary size:", len(vocab))

# ==================================
# 4. ENCODE TEXT
# ==================================

encode = [
    word_to_id[word]
    for word in words
]

# ==================================
# 5. CREATE INPUT AND TARGET SEQUENCES
# ==================================

sequence_length = 5

X = []
Y = []

for i in range(len(encode) - sequence_length):
    X.append(encode[i:i + sequence_length])
    Y.append(encode[i + sequence_length])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("First input:",
      [id_to_word[i.item()] for i in X[0]])

print("First target:", id_to_word[Y[0].item()])

# ==================================
# 6. BUILD GRU MODEL
# ==================================

class GRULanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, hidden = self.gru(embedded)

        logits = self.fc(output[:, -1, :])

        return logits

# ==================================
# 7. INITIALIZE MODEL
# ==================================

model = GRULanguageModel(
    vocab_size=len(vocab),
    embedding_dim=32,
    hidden_size=64
)

print(model)

# ==================================
# 8. LOSS AND OPTIMIZER
# ==================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==================================
# 9. TRAINING
# ==================================

epochs = 300

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predictions = model(X)

    loss = criterion(predictions, Y)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss.item():.4f}"
        )

print("\nTraining completed!")

print("\nTraining completed!")

# ==================================
# 10. SHOW ACTUAL PREDICTIONS
# ==================================

model.eval()
with torch.no_grad():
    predictions = model(X)
    predicted_ids = predictions.argmax(dim=1)
    print("\nPREDICTION RESULTS")
    print("=" * 60)
    for i in range(15):
        input_words = [
            id_to_word[token.item()]
            for token in X[i]
        ]
        actual_word = id_to_word[Y[i].item()]
        predicted_word = id_to_word[predicted_ids[i].item()]
        print("Input:     ", " ".join(input_words))
        print("Actual:    ", actual_word)
        print("Predicted: ", predicted_word)
        print("-" * 60)

