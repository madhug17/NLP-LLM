import numpy as np
sentences = [
    "machine learning is powerful",
    "deep learning is interesting"
]
vocabulary = sorted(set(
    word
    for sentence in sentences
    for word in sentence.split()
))
print("Vocabulary:",vocabulary)

word_to_index = {
    word:index
    for index,word in enumerate(vocabulary)
}
print("Indexing to words:",word_to_index)

for sentence in sentences:
    print("\nSentence:", sentence)
    for wor in sentence.split():
        vector = np.zeros(len(vocabulary))
        vector[word_to_index[wor]] = 1
        print(wor, ":", vector)