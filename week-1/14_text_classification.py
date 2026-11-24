from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

labels = [
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
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences)
print("TF-IDF shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
model = LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)

# 6. Evaluate model
print("\nActual labels:", y_test)
print("Predictions:", predictions)

accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Predict a new sentence
new_sentence = ["artificial intelligence is improving software"]

new_vector = tfidf.transform(new_sentence)
result = model.predict(new_vector)

print("\nNew Sentence:", new_sentence[0])
print("Predicted Category:", result[0])
