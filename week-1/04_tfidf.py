## TF-IDF
import math
documents = [
    "python is a programming language",
    "python is used for machine learning",
    "machine learning is interesting"
]
tokenized_docs = [
    doc.lower().split()
    for doc in documents
]
def calculate_tf(document):
    tf = {} # Term Frequency
    total_words = len(document)
    for word in document:
        tf[word] = document.count(word)/total_words
    return tf
for doc in tokenized_docs:
    print(calculate_tf(doc))

def calculate_idf(tokenized_docs):
    idf = {}
    total_docs = len(tokenized_docs)
    vocal = set(
        word
        for doc in tokenized_docs
        for word in doc
    )
    for word in vocal:
        doc_fre= sum(
            1 for doc in tokenized_docs
            if word in doc
        )
        idf[word] = math.log(
            total_docs/doc_fre
        )
    return idf
        
idf_values = calculate_idf(tokenized_docs)
print("\nIDF values:")
print(idf_values)
tfidf_matrix = []
for doc in tokenized_docs:
    tf = calculate_tf(doc)
    row = {}
    for word in idf_values:
        row[word] = tf.get(word,0)*idf_values[word]
    tfidf_matrix.append(row)
print("\nTF-IDF Matrix:")

for row in tfidf_matrix:
    print(row)
