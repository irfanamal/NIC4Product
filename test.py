import os
import sys
import nltk
import torch
import pickle
import time
import pandas as pd
import numpy
from torchvision import transforms
from dataset.Vocabulary import Vocabulary
from dataLoader import getLoader
from model.model3 import EncoderCNN, DecoderLSTM

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
vocabulary_path = os.path.join(path, 'dataset', 'vocab.pkl')
title_path = os.path.join(path, 'dataset', 'test_set.csv')
image_path = os.path.join(path, 'dataset', 'test_set')
encoder_path = os.path.join(path, 'trained', 'EncoderCNN', 'v13', 'encoder-7-final.ckpt')
decoder_path = os.path.join(path, 'trained', 'DecoderLSTM', 'v13', 'decoder-7-final.ckpt')
logs_path = os.path.join(path, 'logs', 'v13', 'beam20')
embedding_size = 512
resize = 224
momentum = 0.0001
hidden_size = 1024
num_layers = 1
max_length = 40
batch_size = 1
num_workers = 0
beam_width = 20

#Create transforms
transforms = transforms.Compose([transforms.Resize(resize), transforms.RandomCrop(resize), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#Load vocabulary
with open(vocabulary_path, 'rb') as f:
	vocabulary = pickle.load(f)
print('Vocabulary loaded. Size:', len(vocabulary))

#Prepare titles
titles = pd.read_csv(title_path)
titles = titles['title'].tolist()

#Prepare data loader
data_loader = getLoader(image_path, titles, vocabulary, transforms, batch_size, False, num_workers)

#Setup model
encoder = EncoderCNN(embedding_size, momentum).eval()
decoder = DecoderLSTM(embedding_size, hidden_size, len(vocabulary), num_layers, max_length).eval()
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

#Prepare metric
standalone_bleu1_score = 0
standalone_bleu2_score = 0
standalone_bleu3_score = 0
standalone_bleu4_score = 0
cumulative_bleu1_score = 0
cumulative_bleu2_score = 0
cumulative_bleu3_score = 0
cumulative_bleu4_score = 0
total_time = 0

#Testing
total_data = len(data_loader)
with torch.no_grad():
	for i, (indexes, images, titles, lengths) in enumerate(data_loader):
			
		images = images.to(device)
		start = time.time()
		features = encoder(images)

		sampled_ids, probs = decoder.beamSearch(features, beam_width)
		# sampled_ids = decoder.greedySearch(features)

		end = time.time()
		elapsed = end - start

		sentence = []
		for j in range(beam_width):
			sentence.append(sampled_ids[j][0].cpu().numpy())
		# sampled_ids = sampled_ids[0].cpu().numpy()

		titles = titles.detach().cpu().numpy()
		ground_truth = []
		for title in titles:
			ground_truth.append(idToWord(title))

		generated = []
		for j in range(beam_width):
			generated.append(idToWord(sentence[j]))
		# generated = idToWord(sampled_ids)
		
		STANDALONE_BLEU1 = 0
		STANDALONE_BLEU2 = 0
		STANDALONE_BLEU3 = 0
		STANDALONE_BLEU4 = 0
		CUMULATIVE_BLEU1 = 0
		CUMULATIVE_BLEU2 = 0
		CUMULATIVE_BLEU3 = 0
		CUMULATIVE_BLEU4 = 0

		if len(generated[0]) > 1:
			STANDALONE_BLEU1 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(1., 0, 0, 0),smoothing_function=chencherry.method7)
			STANDALONE_BLEU2 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(0, 1., 0, 0),smoothing_function=chencherry.method7)
			STANDALONE_BLEU3 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(0, 0, 1., 0),smoothing_function=chencherry.method7)
			STANDALONE_BLEU4 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(0, 0, 0, 1.),smoothing_function=chencherry.method7)
			CUMULATIVE_BLEU1 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(1./1., 0, 0, 0),smoothing_function=chencherry.method7)
			CUMULATIVE_BLEU2 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(1./2., 1./2., 0, 0),smoothing_function=chencherry.method7)
			CUMULATIVE_BLEU3 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(1./3., 1./3., 1./3., 0),smoothing_function=chencherry.method7)
			CUMULATIVE_BLEU4 = nltk.translate.bleu_score.sentence_bleu(ground_truth,generated[0],weights=(1./4., 1./4., 1./4., 1./4.),smoothing_function=chencherry.method7)

		standalone_bleu1_score += STANDALONE_BLEU1
		standalone_bleu2_score += STANDALONE_BLEU2
		standalone_bleu3_score += STANDALONE_BLEU3
		standalone_bleu4_score += STANDALONE_BLEU4
		cumulative_bleu1_score += CUMULATIVE_BLEU1
		cumulative_bleu2_score += CUMULATIVE_BLEU2
		cumulative_bleu3_score += CUMULATIVE_BLEU3
		cumulative_bleu4_score += CUMULATIVE_BLEU4
		total_time += elapsed

		with open(os.path.join(logs_path, 'all_testing_result.txt'), 'a') as f:
			f.write('File Number: ' + str(indexes[0]) + '\n')
			f.write('Ground truth: ' + str(ground_truth[0]) + '\n')
			for j in range(beam_width):
				f.write('Generated ' + str(j+1) + ': ' + str(generated[j]) + '\n')
			for j in range(beam_width):
				f.write('Probability ' + str(j+1) + ': ' + str(probs[j].item()) + '\n')
			# f.write('Generated: ' + str(generated) + '\n')
			f.write('STANDALONE BLEU1: ' + str(STANDALONE_BLEU1) + '\n')
			f.write('STANDALONE BLEU2: ' + str(STANDALONE_BLEU2) + '\n')
			f.write('STANDALONE BLEU3: ' + str(STANDALONE_BLEU3) + '\n')
			f.write('STANDALONE BLEU4: ' + str(STANDALONE_BLEU4) + '\n')
			f.write('CUMULATIVE BLEU1: ' + str(CUMULATIVE_BLEU1) + '\n')
			f.write('CUMULATIVE BLEU2: ' + str(CUMULATIVE_BLEU2) + '\n')
			f.write('CUMULATIVE BLEU3: ' + str(CUMULATIVE_BLEU3) + '\n')
			f.write('CUMULATIVE BLEU4: ' + str(CUMULATIVE_BLEU4) + '\n')
			f.write('Elapsed time: ' + str(elapsed) + '\n\n')

		if i%(total_data//4) == 0:
			print('({}/{}) STANDALONE BLEU1 {}'.format(i,total_data,standalone_bleu1_score/(i+1)))
			print('({}/{}) STANDALONE BLEU2 {}'.format(i,total_data,standalone_bleu2_score/(i+1)))
			print('({}/{}) STANDALONE BLEU3 {}'.format(i,total_data,standalone_bleu3_score/(i+1)))
			print('({}/{}) STANDALONE BLEU4 {}'.format(i,total_data,standalone_bleu4_score/(i+1)))
			print('({}/{}) CUMULATIVE BLEU1 {}'.format(i,total_data,cumulative_bleu1_score/(i+1)))
			print('({}/{}) CUMULATIVE BLEU2 {}'.format(i,total_data,cumulative_bleu2_score/(i+1)))
			print('({}/{}) CUMULATIVE BLEU3 {}'.format(i,total_data,cumulative_bleu3_score/(i+1)))
			print('({}/{}) CUMULATIVE BLEU4 {}'.format(i,total_data,cumulative_bleu4_score/(i+1)))

#Save summary of testing
with open(os.path.join(logs_path, 'avg_testing_inference.txt'), 'a') as f:
	f.write('Average Standalone BLEU-1: ' + str(standalone_bleu1_score/total_data) + '\n')
	f.write('Average Standalone BLEU-2: ' + str(standalone_bleu2_score/total_data) + '\n')
	f.write('Average Standalone BLEU-3: ' + str(standalone_bleu3_score/total_data) + '\n')
	f.write('Average Standalone BLEU-4: ' + str(standalone_bleu4_score/total_data) + '\n')
	f.write('Average Cumulative BLEU-1: ' + str(cumulative_bleu1_score/total_data) + '\n')
	f.write('Average Cumulative BLEU-2: ' + str(cumulative_bleu2_score/total_data) + '\n')
	f.write('Average Cumulative BLEU-3: ' + str(cumulative_bleu3_score/total_data) + '\n')
	f.write('Average Cumulative BLEU-4: ' + str(cumulative_bleu4_score/total_data) + '\n')
	f.write('Average Inference Time: ' + str(total_time/total_data) + '\n')