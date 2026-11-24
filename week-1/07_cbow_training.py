sentence = "artificial intelligence is changing the world"
words = sentence.split()
window_size = 1
training_data = []
for i , target in enumerate(words):
    context = []
    start = max(0,i-window_size)
    end = min(len(words),i+window_size+1)
    for j in range(start,end):
        if j != i:
            context.append(words[j])
    training_data.append((context, target))
print(training_data)