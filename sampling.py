import os
import sys
import torch
import pickle
import time
import numpy
import nltk
import urllib.request
from dataset.Vocabulary import Vocabulary
from torchvision import transforms
from PIL import Image
from model.model import EncoderCNN, DecoderLSTM

def idToWord(sentence):
	s = []
	for wordID in sentence:
		word = vocabulary.idx2word[wordID]
		s.append(word)
	return s

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = sys.path[0]

vocabulary_path = os.path.join(path, 'dataset', 'vocab.pkl')
image_path = os.path.join(path, 'dataset', 'gambar_nyoba')
encoder_path = os.path.join(path, 'trained', 'EncoderCNN', 'v12', 'encoder-8-final.ckpt')
decoder_path = os.path.join(path, 'trained', 'DecoderLSTM', 'v12', 'decoder-8-final.ckpt')
embedding_size = 512
momentum = 0.0001
hidden_size = 1024
num_layers = 1
max_length = 40
batch_size = 1
num_workers = 0
resize = 224

transforms = transforms.Compose([transforms.Resize(resize), transforms.RandomCrop(resize), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with open(vocabulary_path, 'rb') as f:
	vocabulary = pickle.load(f)

encoder = EncoderCNN(embedding_size, momentum).eval()
decoder = DecoderLSTM(embedding_size, hidden_size, len(vocabulary), num_layers, max_length).eval()
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

with torch.no_grad():
	while True:
		source = input('Input the source (online/offline): ')
		file_path = ''
		if source == 'offline':
			filename = input('Input the image's file name: ')
			file_path = os.path.join(image_path, filename)
		else:
			images = os.listdir(image_path)
			i = 0
			while True:
				if str(i)+'.jpg' not in images:
					break
				else:
					i += 1
			filename = str(i)+'.jpg'
			url = input ('Input the image's URL: ')
			file_path = os.path.join(image_path, filename)
			urllib.request.urlretrieve(url, file_path)
			print('Gambar berhasil didownload')
			print('Nama file:', str(i)+'.jpg')
		image = Image.open(file_path).convert('RGB')
		image = transforms(image).unsqueeze(0)

		image = image.to(device)

		# print('Start')
		# start = time.time()

		feature = encoder(image)
		sampled_ids = decoder.greedySearch(feature)

		# end = time.time()
		# print('Finish')
		elapsed = end - start

		sampled_ids = sampled_ids[0].cpu().numpy()
		sentence = idToWord(sampled_ids)

		generated = ''
		for i in range(1,len(sampled_ids)-1):
			generated += sentence[i]
			if i < len(sampled_ids)-2:
				generated += ' '
		print(generated)
		# print(elapsed)