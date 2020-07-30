import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import torch
from dataset.Vocabulary import Vocabulary
from PIL import Image

class Dataset(data.Dataset):

	def __init__(self, image_path, titles, vocab, transform=None):
		
		self.image_path = image_path
		self.titles = titles
		self.vocab = vocab
		self.transform = transform

	def __getitem__(self, index):
		
		image = Image.open(os.path.join(self.image_path, str(index)+'.jpg')).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		caption = []
		caption.append(self.vocab('<s>'))
		tokens = nltk.tokenize.word_tokenize(self.titles[index].lower())
		caption.extend([self.vocab(token) for token in tokens])
		caption.append(self.vocab('</s>'))
		title = torch.Tensor(caption)

		return index, image, title

	def __len__(self):
		return len(self.titles)

def collate_fn(data):

	data.sort(key=lambda x: len(x[2]), reverse=True)
	indexes, images, captions = zip(*data)

	indexes = list(indexes)
	# print(indexes)

	images = torch.stack(images, 0)

	lengths = [len(caption) for caption in captions]
	titles = torch.zeros(len(captions), max(lengths)).long()
	for i, caption in enumerate(captions):
		length = lengths[i]
		titles[i, :length] = caption[:length]

	return indexes, images, titles, lengths

def getLoader(image_path, titles, vocab, transform, batch_size, shuffle, num_workers):

	dataset = Dataset(image_path, titles, vocab, transform)

	data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

	return data_loader
