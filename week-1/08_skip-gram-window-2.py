sentence = "artificial intelligence is changing the world"
window_size = 2
words = sentence.split()
training_data = []
for i, input_word in enumerate(words):
    start = max(0,i-window_size)
    end = min(len(words),i+window_size+1)
    for j in range(start,end):
        if j != i:
            training_data.append((input_word, words[j]))
print(training_data)