import os
import sys
import nltk
import pickle
import pandas as pd
from dataset.Vocabulary import Vocabulary

path = sys.path[0]

# with open(os.path.join(path, 'dataset', 'vocabulary_train_unprocessed.pkl'), 'rb') as f:
# 	ayam = pickle.load(f)
# print(len(ayam))

dataset = pd.read_csv(os.path.join(path, 'dataset', 'train_set_raw.csv'))


# print(dataset.shape[0])

# dataset_id = []

# file_id = dataset['file_id'].tolist()
# for i in file_id:
# 	dataset_id.append(int(i))

# dataset_id = set(dataset_id)
# dataset_id = list(dataset_id)
# print(len(dataset_id))

# for i in range(len(dataset_id)-1):
# 	if dataset_id[i] != dataset_id[i+1]-1:
# 		print("Ga dense cek index ke -", i)

vocabulary = Vocabulary()
vocabulary.addWord('<p>')
vocabulary.addWord('<s>')
vocabulary.addWord('</s>')
vocabulary.addWord('<u>')
dataset = dataset['title'].tolist()

# vocab = []
# counter = []

for title in dataset:
	words = nltk.tokenize.word_tokenize(title.lower())
	for word in words:
		vocabulary.addWord(word)

print(len(vocabulary))
with open(os.path.join(path, 'dataset', 'vocab_raw.pkl'), 'wb') as f:
	pickle.dump(vocabulary, f)
		# if word not in vocab:
			# vocab.append(word)
			# counter.append(1)
# 		else:
# 			counter[vocab.index(word)] += 1

# vocab_counter = []
# for i in range(len(vocab)):
# 	vocab_counter.append((vocab[i],counter[i]))

# vocab_counter = sorted(vocab_counter, key=lambda tup: tup[0])

# # print(vocab[4584])

# with open(os.path.join(path, 'vocabulary.txt'), 'a') as f:
# 	for i in range(len(vocab_counter)):
# 		f.write(vocab_counter[i][0] + ' ' + str(vocab_counter[i][1]) + '\n')
