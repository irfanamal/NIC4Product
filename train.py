import torch
import os
import sys
import pandas as pd
import pickle
import torch.nn as nn
import time
import nltk
import numpy
from torchvision import transforms
from dataLoader import getLoader
from dataset.Vocabulary import Vocabulary
from model.model import EncoderCNN, DecoderLSTM
from torch.nn.utils.rnn import pack_padded_sequence

def idToWord(sentence):
	s = []
	for wordID in sentence:
		word = vocabulary.idx2word[wordID]
		s.append(word)
		if word == '</s>':
			break
	return s

#Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chencherry = nltk.translate.bleu_score.SmoothingFunction()
path = sys.path[0]

#Hyperparameter
encoder_path = os.path.join(path, 'trained', 'EncoderCNN', 'v12')
decoder_path = os.path.join(path, 'trained', 'DecoderLSTM', 'v12')
trained_encoder = os.path.join(encoder_path, 'encoder-20-final.ckpt')
trained_decoder = os.path.join(decoder_path, 'decoder-20-final.ckpt')
vocabulary_path = os.path.join(path, 'dataset', 'vocab.pkl')
image_path = os.path.join(path, 'dataset', 'train_set')
title_path = os.path.join(path, 'dataset', 'train_set.csv')
val_image_path = os.path.join(path, 'dataset', 'val_set')
val_title_path = os.path.join(path, 'dataset', 'val_set.csv')
logs_path = os.path.join(path, 'logs', 'v12')
embedding_size = 512
momentum = 0.0001
learning_rate = 0.0001
hidden_size = 1024
num_layers = 1
resize = 224
max_length = 40
batch_size = 8
num_workers = 0
num_epochs = 26
step = 5

#Create folders
if not os.path.exists(encoder_path):
	os.makedirs(encoder_path)
if not os.path.exists(decoder_path):
	os.makedirs(decoder_path)
if not os.path.exists(logs_path):
	os.makedirs(logs_path)

#Create trasforms
transforms = transforms.Compose([transforms.Resize(resize), transforms.RandomCrop(resize), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#Load vocabulary
with open(vocabulary_path, 'rb') as f:
	vocabulary = pickle.load(f)
print('Vocabulary loaded. Size:', len(vocabulary))

#Prepare product title list
train_titles = pd.read_csv(title_path)
train_titles = train_titles['title'].tolist()
val_titles = pd.read_csv(val_title_path)
val_titles = val_titles['title'].tolist()

#Prepare data loader
data_loader = getLoader(image_path, train_titles, vocabulary, transforms, batch_size, True, num_workers)
val_loader = getLoader(val_image_path, val_titles, vocabulary, transforms, batch_size, True, num_workers)
bleu_train_loader = getLoader(image_path, train_titles, vocabulary, transforms, 1, False, num_workers)
bleu_val_loader = getLoader(val_image_path, val_titles, vocabulary, transforms, 1, False, num_workers)

#Create model
encoder = EncoderCNN(embedding_size, momentum).to(device)
decoder = DecoderLSTM(embedding_size, hidden_size, len(vocabulary), num_layers, max_length).to(device)

encoder.load_state_dict(torch.load(trained_encoder))
decoder.load_state_dict(torch.load(trained_decoder))

#Configure optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

#Training
total_step = len(data_loader)
total_val = len(val_loader)
total_bleu_train = len(bleu_train_loader)
total_bleu_val = len(bleu_val_loader)

for epoch in range(21,num_epochs):
	loss_train = 0
	loss_val = 0
	encoder.train()
	decoder.train()
	
	print('Start Train')
	for i, (indexes, images, titles, lengths) in enumerate(data_loader):

		images = images.to(device)
		titles = titles.to(device)
		targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

		features = encoder(images)
		outputs = decoder(features, titles, lengths)

		loss = criterion(outputs, targets)
		decoder.zero_grad()
		encoder.zero_grad()
		loss.backward()
		optimizer.step()

		loss_train += loss.item() * images.size(0)

		if i % (len(data_loader)//4) == 0:
			print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch, num_epochs, i, total_step, loss.item()))
		
	torch.save(decoder.state_dict(), os.path.join(decoder_path, 'decoder-{}-final.ckpt'.format(epoch)))
	torch.save(encoder.state_dict(), os.path.join(encoder_path, 'encoder-{}-final.ckpt'.format(epoch)))

	with open(os.path.join(logs_path, 'train_epoch_loss.txt'), 'a') as f:
		f.write(str(loss_train/len(train_titles)) + '\n')

	with torch.no_grad():
		encoder.eval()
		decoder.eval()

		print('Start Validate')
		for i, (indexes, images, titles, lengths) in enumerate(val_loader):
			
			images = images.to(device)
			titles = titles.to(device)
			targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

			features = encoder(images)
			outputs = decoder(features, titles, lengths)

			loss = criterion(outputs, targets)

			loss_val += loss.item() * images.size(0)

			if i % (len(val_loader)//4) == 0:
				print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}'.format(epoch, num_epochs, i, total_val, loss.item()))

		with open(os.path.join(logs_path, 'val_epoch_loss.txt'), 'a') as f:
			f.write(str(loss_val/len(val_titles)) + '\n')

		print('Epoch [{}/{}], Epoch Train Loss: {:.4f}, Epoch Validation Loss: {:.4f}'.format(epoch, num_epochs, loss_train/len(train_titles), loss_val/len(val_titles)))

		if epoch % step == 0:
			print('Calculate BLEU Train')
			bleu1_train_score = 0
			for i, (indexes, images, titles, lengths) in enumerate(bleu_train_loader):

				images = images.to(device)
				
				features = encoder(images)
				sampled_ids = decoder.greedySearch(features)

				sampled_ids = sampled_ids[0].cpu().numpy()

				titles = titles.detach().cpu().numpy()
				ground_truth = []
				for title in titles:
					ground_truth.append(idToWord(title))

				generated = idToWord(sampled_ids)

				temp_score = 0
				if len(generated) > 1:
					temp_score = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated,weights=(1., 0, 0, 0),smoothing_function=chencherry.method7)
				bleu1_train_score += temp_score

				if i % (len(bleu_train_loader)//4) == 0:
					print('Epoch [{}/{}], Step [{}/{}], BLEU-1 Train: {:.4f}'.format(epoch, num_epochs, i, total_bleu_train, temp_score))

			avg_bleu1_train = bleu1_train_score/total_bleu_train
			with open(os.path.join(logs_path, 'train_epoch_bleu.txt'), 'a') as f:
				f.write(str(avg_bleu1_train) + '\n')

			print('Calculate BLEU Validation')
			bleu1_val_score = 0
			for i, (indexes, images, titles, lengths) in enumerate(bleu_val_loader):

				images = images.to(device)
				
				features = encoder(images)
				sampled_ids = decoder.greedySearch(features)

				sampled_ids = sampled_ids[0].cpu().numpy()

				titles = titles.detach().cpu().numpy()
				ground_truth = []
				for title in titles:
					ground_truth.append(idToWord(title))

				generated = idToWord(sampled_ids)

				temp_score = 0
				if len(generated) > 1:
					temp_score = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated,weights=(1., 0, 0, 0),smoothing_function=chencherry.method7)
				bleu1_val_score += temp_score

				if i % (len(bleu_val_loader)//4) == 0:
					print('Epoch [{}/{}], Step [{}/{}], BLEU-1 Validation: {:.4f}'.format(epoch, num_epochs, i, total_bleu_val, temp_score))

			avg_bleu1_val = bleu1_val_score/total_bleu_val
			with open(os.path.join(logs_path, 'val_epoch_bleu.txt'), 'a') as f:
				f.write(str(avg_bleu1_val) + '\n')

			print('Epoch [{}/{}], Epoch Train BLEU-1: {:.4f}, Epoch Validation BLEU-1: {:.4f}'.format(epoch, num_epochs, avg_bleu1_train, avg_bleu1_val))
