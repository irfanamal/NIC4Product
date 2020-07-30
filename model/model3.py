import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):

	def __init__(self, embedding_size, momentum):

		super(EncoderCNN, self).__init__()
		googlenet = models.googlenet(pretrained=True)
		modules = list(googlenet.children())[:-1]
		self.googlenet = nn.Sequential(*modules)
		self.linear = nn.Linear(googlenet.fc.in_features, embedding_size)
		self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=momentum)

	def forward(self, images):

		with torch.no_grad():
			features = self.googlenet(images)
		features = torch.flatten(features,1)
		features = self.linear(features)
		features = self.batch_norm(features)

		return features

class DecoderLSTM(nn.Module):
	def __init__(self, embedding_size, hidden_size, vocabulary_size, num_layers, max_length):

		super(DecoderLSTM, self).__init__()
		self.word_embedding = nn.Embedding(vocabulary_size, embedding_size)
		self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocabulary_size)
		self.max_length = max_length

	def forward(self, features, titles, lengths):

		embeddings = self.word_embedding(titles)
		embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
		sequence = pack_padded_sequence(embeddings, lengths, batch_first=True) 
		hiddens, _ = self.lstm(sequence)
		outputs = self.linear(hiddens[0])
        
		return outputs

	def greedySearch(self, features):
		states = None
		sampled_ids = []

		inputs = features.unsqueeze(1)
		for i in range(self.max_length):
			hiddens, states = self.lstm(inputs, states)
			outputs = self.linear(hiddens.squeeze(1))
			_, word_id = outputs.max(1)
			sampled_ids.append(word_id)
			if word_id[0] == 2:
				break
			else:
				inputs = self.word_embedding(word_id)
				inputs = inputs.unsqueeze(1)

		sampled_ids = torch.stack(sampled_ids, 1)
		return sampled_ids

	def beamSearch(self, features, width):
		states = [None for i in range(width)]
		sampled_ids = [[] for i in range(width)]
		probs = [1 for i in range(width)]
		# print(probs)
		# print(sampled_ids)

		inputs = features.unsqueeze(1)
		hiddens, newStates = self.lstm(inputs, None)
		outputs = self.linear(hiddens.squeeze(1))
		outputs = nn.functional.softmax(outputs, 1)
		for i in range(width):
			prob, word_id = outputs.max(1)
			states[i] = newStates
			sampled_ids[i].append(word_id)
			probs[i] *= prob[0]
			outputs[0][word_id[0]] = 0
		# print('Initial Probs', probs)
		# print('Initial Sentences', sampled_ids)
		# print('Initial States', states[:width][:3])
		# print('')
		# print(sampled_ids)

		for i in range(1,self.max_length):
			temp_states = [None for j in range(width)]
			temp_sampled_ids = [[] for j in range(width)]
			temp_probs = [[probs[j].clone() for k in range(width)] for j in range(width)]
			# print(sampled_ids)

			for j in range(width):
				# print(temp_probs)
				# if sampled_ids[j][-1][0] == 2:
				# 	temp_states[j] = None
				# 	for k in range(width):
				# 		temp_sampled_ids[j].append(sampled_ids[j][-1])
				# 		if i == 1:
				# 			temp_probs[j][k] *= 0
				# 		else:
				# 			if k == 0:
				# 				temp_probs[j][k] *= 1
				# 			else:
				# 				temp_probs[j][k] *= 0
				# else:
				inputs = self.word_embedding(sampled_ids[j][-1])
				inputs = inputs.unsqueeze(1)
				hiddens, newStates = self.lstm(inputs, states[j])
				temp_states[j] = newStates
				outputs = self.linear(hiddens.squeeze(1))
				outputs = nn.functional.softmax(outputs, 1)
				for k in range(width):
					prob, word_id = outputs.max(1)
					temp_sampled_ids[j].append(word_id)
					# print(temp_sampled_ids)
					# print(temp_probs[j])
					# print(prob[0])
					temp_probs[j][k] *= prob[0]
					# print(temp_probs[j])
					outputs[0][word_id[0]] = 0
				# print(temp_probs)

			max_column = []
			max_row = []
			max_prob = []

			# print('Current Probs', temp_probs)
			# print('Current Words', temp_sampled_ids)
			# print('Current States', temp_states[:width][:3])
			# print('')

			for j in range(width):
				prob = 0
				row = 0
				column = 0
				for k in range(width):
					for l in range(width):
						if temp_probs[k][l] > prob:
							prob = temp_probs[k][l].clone()
							row = k
							column = l
				max_prob.append(prob)
				max_row.append(row)
				max_column.append(column)
				temp_probs[row][column] = 0
				
			new_probs = max_prob
			# print('New Probs', new_probs)
			new_sampled_ids = []
			# print(new_sampled_ids)
			new_states = []
			for j in range(width):
				new_sampled_ids.append(sampled_ids[max_row[j]].copy())
				# print(new_sampled_ids)
				new_sampled_ids[j].append(temp_sampled_ids[max_row[j]][max_column[j]])
				# print(new_sampled_ids)
				# print('')
				new_states.append(temp_states[max_row[j]])
			states = new_states
			sampled_ids = new_sampled_ids
			probs = new_probs
			# print('New Probs', probs)
			# print('New Sentences', sampled_ids)
			# print('New States', states[:width][:3])
			# print('')

			count = 0
			for j in range(width):
				for k in range(len(sampled_ids[j])):
					if sampled_ids[j][k][0] == 2:
						count += 1
						break
				# if torch.tensor([2]).cuda() in sampled_ids[j]:
				# 	count += 1
			if count == width:
				break

		result = []
		for i in range(width):
			result.append(torch.stack(sampled_ids[i], 1))
		# print('\n')
		return result, probs