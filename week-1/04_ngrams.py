sentences = [
    "machine learning is powerful",
    "deep learning is interesting",
    "machine learning is interesting"
]
def generate_ngrams(sentence,n):
    words = sentence.lower().split()
    ngram = []
    for i in range(len(words)-n+1):
        gram = words[i:i+n]
        ngram.append(''.join(gram))
    return ngram
sentence = "machine learning is powerful"

print("Unigrams:", generate_ngrams(sentence, 1))
print("Bigrams:", generate_ngrams(sentence, 2))
print("Trigrams:", generate_ngrams(sentence, 3))


## same with sklearn
from sklearn.feature_extraction.text import CountVectorizer
sentences = [
    "machine learning is powerful",
    "deep learning is interesting",
    "machine learning is interesting"
]
uni = CountVectorizer(ngram_range=(1,1))
uni_matrix = uni.fit_transform(sentences)
print("Unigram vocabulary:")
print(uni.get_feature_names_out())
print(uni_matrix.toarray())

# Bigrams
bi = CountVectorizer(ngram_range=(2, 2))
bi_matrix = bi.fit_transform(sentences)

print("\nBigram vocabulary:")
print(bi.get_feature_names_out())
print(bi_matrix.toarray())

## Trigram
tri = CountVectorizer(ngram_range=(3,3))
tri_matrix = tri.fit_transform(sentences)
print("\nTrigram vocabulary:")
print(tri.get_feature_names_out())
print(tri_matrix.toarray())