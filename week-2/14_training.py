import torch
import torch.nn as nn
with open("ai_ml_corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()
words= text.lower().split()
print("Total characters:", len(text))
print("Total words:", len(words))
print("First 30 words:", words[:30])

import string
clean_text = text.lower().translate(
    str.maketrans("", "", string.punctuation)
)
words = clean_text.split()
vocab = sorted(set(words))
word_to_id = {
    word:index
    for index,word in enumerate(vocab)
}
id_to_word = {
    index:word
    for word,index in word_to_id.items()
}
encode = [
    word_to_id[word] for word in words
]
sequence_lenght = 5
x = []
y = []

for i in range(len(encode)-sequence_lenght):
    x.append(encode[i:i+sequence_lenght])
    y.append(encode[i+sequence_lenght])
x = torch.tensor(x,dtype=torch.long)
y = torch.tensor(y,dtype=torch.long)

class GRULanguageModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.gru =nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,vocab_size)
    def forward(self,x):
        embedded = self.embedding(x)
        output,hidden = self.gru(embedded)
        logits = self.fc(output[:, -1, :])
        return logits
    