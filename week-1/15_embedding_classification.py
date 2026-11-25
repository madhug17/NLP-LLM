import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec

sentences = [
    "machine learning is transforming technology",
    "deep learning uses neural networks",
    "python is a popular programming language",
    "artificial intelligence is growing rapidly",
    "software developers build applications",

    "football is a popular sport",
    "cricket players hit the ball",
    "the team won the championship",
    "basketball players score points",
    "the player scored a goal"
]
labels =[
    "technology",
    "technology",
    "technology",
    "technology",
    "technology",
    "sports",
    "sports",
    "sports",
    "sports",
    "sports"
]

tokenization = [
    sentence.lower().split()
    for sentence in sentences
]
word2Vec_model = Word2Vec(
    sentences=tokenization,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1,
    workers=1,
    epochs=100

)

def sentence_embedding(sentence):
    vector = [
        word2Vec_model.wv[word]
        for word in sentence
        if word in word2Vec_model.wv
    ]
    if len(vector) == 0:
        return np.zeros(100)
    return np.mean(vector,axis=0)
x = np.array([
    sentence_embedding(sentence)
    for sentence in tokenization
])
y = np.array(labels)
print("Embedding matrix shape:", x.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
# 9. Predict new sentence
new_sentence = "artificial intelligence and machine learning"

new_tokens = new_sentence.lower().split()

new_vector = sentence_embedding(new_tokens).reshape(1, -1)

result = model.predict(new_vector)

print("\nNew sentence:", new_sentence)
print("Predicted category:", result[0])

