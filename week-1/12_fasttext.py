from gensim.models import FastText
sentences =  [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning uses algorithms",
    "python is used for programming",
    "artificial intelligence uses machine learning",
    "programmers are programming applications"
]
tokenization = [
    sentence.lower().split()
    for sentence in sentences
]
fasttext_model = FastText(
    sentences=tokenization,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    min_n=3,
    max_n=5,
    workers=1,
    epochs=100
)
vector = fasttext_model.wv["programming"]
print("Embedding:")
print(vector)
print("\nEmbedding shape:")
print(vector.shape)
print("\nSimilar words:")
print(fasttext_model.wv.most_similar("programming",topn = 5))
print("\nUnseen word embedding:")
print(fasttext_model.wv["programmerxyz"])