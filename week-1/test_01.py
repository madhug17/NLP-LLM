import numpy as np 
sentences = [
    "python is an amazing programming language",
    "machine learning uses python",
    "deep learning uses neural networks",
    "python is popular for machine learning",
    "neural networks are powerful",
    "artificial"
]
vocal = sorted(set(
    word
    for sentence in sentences
    for word in sentence.split()
))
word_to_text = {
    word:index
    for index,word in enumerate(vocal)
}
for sen in sentences:
    for word in sen.split():
        vector = np.zeros(len(vocal))
        vector[word_to_text[word]] = 1
        print(word, ":", vector)

## BoW Matrix
text = []
for sent in sentences:
    words = sent.split()
    vector = np.zeros(len(vector),dtype=int)
    for word in words:
        index = vocal.index(word)
        vector[index]+=1
    text.append(vector)
bow_matrxi = np.array(text)
print("\nBag of Words Matrix:")
print(text)

# by using the sklearn
from sklearn.feature_extraction.text import CountVectorizer
vectors = CountVectorizer()
x = vectors.fit_transform(sentences)
print(vectors.get_feature_names_out())
print("\nBoW Matrix:")
print(x.toarray())