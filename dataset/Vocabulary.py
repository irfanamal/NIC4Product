class Vocabulary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def addWord(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):
		if word not in self.word2idx:
			return self.word2idx['<u>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)