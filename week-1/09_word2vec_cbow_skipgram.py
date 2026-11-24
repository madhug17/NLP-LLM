from gensim.models import Word2Vec

sentences = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning uses algorithms",
    "neural networks are used in deep learning",
    "python is used for machine learning",
    "artificial intelligence uses machine learning",
    "python is a programming language",
    "deep learning is a part of artificial intelligence"
]

tokenization = [
    sentence.lower().split()
    for sentence in sentences
]
cbow_models = Word2Vec( # it will find the central word 
    sentences=tokenization,
    window =2,
    min_count = 1,
    sg = 0,
    workers = 1,
    epochs = 100,
    vector_size = 100
)
skipgram_model = Word2Vec( # it will check the surrounding words 
    sentences=tokenization,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    workers=1,
    epochs=100
)
print("CBOW embedding:")
print(cbow_models.wv["machine"])

print("\nSkip-gram embedding:")
print(skipgram_model.wv["machine"])

print("\nCBOW similar words:")
print(cbow_models.wv.most_similar("deep", topn=5))
print("\nSkip-gram similar words:")
print(skipgram_model.wv.most_similar("deep", topn=5))