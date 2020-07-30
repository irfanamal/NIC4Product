import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):

	def __init__(self, embedding_size, momentum):

		super(EncoderCNN, self).__init__()
		mobile_net = models.mobilenet_v2(pretrained=True)
		modules = list(mobile_net.children())[:-1]
		self.mobilenet = nn.Sequential(*modules)
		self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(mobile_net.last_channel, embedding_size))
		self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=momentum)

	def forward(self, images):

		with torch.no_grad():
			features = self.mobilenet(images)
		features = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
		features = self.classifier(features)
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
