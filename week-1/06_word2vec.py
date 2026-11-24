
from gensim.models import Word2Vec

sentences = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning uses algorithms",
    "neural networks are used in deep learning",
    "python is used for machine learning",
    "artificial intelligence uses machine learning"
]

tokenization = [
    sentence.lower().split()
    for sentence in sentences
]

# CBOW
cbow_model = Word2Vec(
    sentences=tokenization,
    vector_size=100,
    window=2,
    min_count=1,
    sg=0,
    workers=1,
    epochs=100
)

# Get embedding
vector = cbow_model.wv["machine"]

print("Machine embedding:")
print(vector)

print("Vector dimensions:")
print(vector.shape)

# Find similar words
similar_words = cbow_model.wv.most_similar("learning", topn=5)


print("\nSimilar words:")
print(similar_words)
