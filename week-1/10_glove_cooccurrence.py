import numpy as np
sentences =[
    "machine learning is powerful",
    "deep learning is useful",
    "machine learning uses python"
]
tokenized = [
    sentence.lower().split()
    for sentence in sentences
]
vocab = sorted(set(
    word
    for sentence in tokenized
    for word in sentence
))
word_to_id = {
    word:index
    for index,word in enumerate(vocab)
}
window_size = 1
co_matrix = np.zeros(
    (
        len(vocab),len(vocab)
    ),dtype=int
)
for sentence in tokenized:
    for i ,centre in enumerate(sentence):
        start = max(0,i-window_size)
        end = min(len(sentence),i+window_size+1)
        for j in range(start,end):
            if j != i:
                context_word = sentence[j]
                center_id = word_to_id[context_word]
                context_id = word_to_id[context_word]
                co_matrix[center_id][context_id] += 1
print("Vocabulary:")
print(vocab)

print("\nWord-to-ID:")
print(word_to_id)

print("\nCo-occurrence Matrix:")
print(co_matrix)

