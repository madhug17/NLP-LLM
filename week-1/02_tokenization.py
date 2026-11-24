import nltk # Natural Language Toolkit
from nltk.tokenize import word_tokenize 
nltk.download("punkt") # we have lot of punkt for situations 
nltk.download("punkt_tab")
text ="""The students are learning advanced machine learning.
They were studying different algorithms and models.
The models are performing exceptionally well!"""
tokens = word_tokenize(text)
print("Original text:", text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

## Charecter split

text = "machine learning"
tokens = list(text)
print(tokens)
print("Total characters:", len(tokens))

## Build our own vocabulary

sentences = [
    "machine learning is powerful",
    "deep learning is interesting",
    "machine learning uses neural networks"
]
vocab = set()
for sentence in sentences:
    for word in sentence.split():
        vocab.add(word)
vocab = sorted(vocab)
word_to_id = {
    word: index
    for index,word in enumerate(vocab)
}
print("Vocabulary:", vocab)
print("Word to ID:", word_to_id)
sentence = "machine learning is powerful"

token_ids = [
    word_to_id[word]
    for word in sentence.split()
]
print("Token IDs:", token_ids)

## Handle unknown tokens (UNK) 
## What happens when a word doesn't exist in our vocabulary?
vocab = {
    "machine": 0,
    "learning": 1,
    "is": 2,
    "powerful": 3,
    "[UNK]": 4
}
sentence = "machine learning is amazing"
tokens = sentence.split()
token_id = [
    vocab.get(word,vocab["[UNK]"])
    for word in tokens
]
print(token_id)

## Padding and truncation
## in this it will make all the matrix in equal lenght with help 
## of appending 0's at end 

sequences = [
    [10, 20, 30],
    [40, 50],
    [60, 70, 80, 90, 100]
]

MAX_LENGTH = 4

processed = []

for sequence in sequences:

    # Truncation
    sequence = sequence[:MAX_LENGTH]

    # Padding
    while len(sequence) < MAX_LENGTH:
        sequence.append(0)

    processed.append(sequence)

for sequence in processed:
    print(sequence)

    
