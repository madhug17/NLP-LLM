import torch
import torch.nn as nn
rnn = nn.RNN(
    input_size=3,
    hidden_size=8,
    batch_first=True
)
x = torch.randn(1,4,3)
output,hidden = rnn(x)
print("Output for first word:",output[:,0,:])
print("Output for second word:", output[:, 1, :])
print("Output for third word:", output[:, 2, :])
print("Output for fourth word:", output[:, 3, :])

print("Final hidden state:", hidden)

# using the tokenization and word_to_id in this 
sentence = "machine learning is powerful"
words = sentence.split()
vocab = sorted(set(words))
word_to_id = {
    word:index
    for index,word in enumerate(vocab)
}
encode = [
    word_to_id[word]
    for word in words
]
print("Vocabulary:", word_to_id)
print("Encoded:", encode)

input_ids = torch.tensor([encode])
embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=5
)
embedded = embedding(input_ids)
print("Input shape:", input_ids.shape)
print("Embedding shape:", embedded.shape)

rnn = nn.RNN(
    input_size=5,
    hidden_size=8,
    batch_first=True
)
output,hidden = rnn(embedded)
print("RNN output shape:", output.shape)
print("Final hidden shape:", hidden.shape)

fc = nn.Linear(8,len(vocab))
predictions = fc(output)
print("Predictions shape:", predictions.shape)