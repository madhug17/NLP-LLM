import gensim.downloader as api
# Load pretrained GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-50") # it is the pre-trained mode in wiki-pedia

#embedding of a word 
vector = glove_model['machine']
print("Embedding:")
print(vector)
print("\nShape:")
print(vector.shape)

# Find similar words
similar = glove_model.most_similar('machine',topn=5)
print("\nSimilar words:")
print(similar)


print("Vocabulary size:", len(glove_model.key_to_index))
print("\nMachine embedding:")
print(glove_model["machine"])

print("\nSimilar words:")
print(glove_model.most_similar('machine',topn=5))
print("\nComputer embedding:")
print(glove_model["computer"])