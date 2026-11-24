import numpy as np
sentences = [
    "machine learning is powerful",
    "deep learning is interesting",
    "machine learning is interesting"
]

vocabulary = sorted(set(
    word 
    for sentence in sentences
    for word in sentence.split()
))
print("Vocabulary:", vocabulary)

word_to_index = {
    word : index
    for index,word in enumerate(vocabulary)
}
print("indexing:",word_to_index)
## creating a document vector
bow_vector = []
for sentence in sentences:
    words = sentence.split()
    vector = np.zeros(len(vocabulary),dtype=int)
    for word in words:
        index = vocabulary.index(word)
        vector[index] += 1
    bow_vector.append(vector)
bow_matrix = np.array(bow_vector)
print("\nBag of Words Matrix:")
print(bow_matrix)

## same with BoW using Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
sentences = [
    "machine learning is powerful",
    "deep learning is interesting",
    "machine learning is interesting"
]
vectorization = CountVectorizer()
x = vectorization.fit_transform(sentences)
print("Vocabulary:")
print(vectorization.get_feature_names_out())
print("\nBoW Matrix:")
print(x.toarray())