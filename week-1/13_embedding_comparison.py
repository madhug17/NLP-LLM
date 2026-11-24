
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import Word2Vec, FastText

# Dataset
sentences = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning uses algorithms",
    "python is used for programming",
    "artificial intelligence uses machine learning"
]

tokenization = [
    sentence.lower().split()
    for sentence in sentences
]

# 1. TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

print("TF-IDF:", tfidf_matrix.shape)

# 2. GloVe
glove = api.load("glove-wiki-gigaword-50")

print("\nGloVe")
print(glove["deep"])
print("Shape:", glove["deep"].shape)
print(glove.most_similar("deep", topn=5))

# 3. FastText
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

print("\nFastText")
vector = fasttext_model.wv["deep"]
print("Embedding:", vector)
print("Shape:", vector.shape)
print(fasttext_model.wv.most_similar("deep", topn=5))

# 4. Word2Vec - CBOW
cbow_model = Word2Vec(
    sentences=tokenization,
    vector_size=100,
    workers=1,
    window=2,
    min_count=1,
    sg=0,
    epochs=100
)

print("\nWord2Vec - CBOW")
vector = cbow_model.wv["deep"]
print("Embedding:", vector)
print("Shape:", vector.shape)
print(cbow_model.wv.most_similar("deep", topn=5))

# 5. Word2Vec - Skip-gram
skipgram_model = Word2Vec(
    sentences=tokenization,
    vector_size=100,
    workers=1,
    window=2,
    min_count=1,
    sg=1,
    epochs=100
)

print("\nWord2Vec - Skip-gram")
vector = skipgram_model.wv["deep"]
print("Embedding:", vector)
print("Shape:", vector.shape)
print(skipgram_model.wv.most_similar("deep", topn=5))
