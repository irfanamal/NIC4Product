import matplotlib.pyplot as plt
import os
import sys

path = sys.path[0]
version = input()

with open(os.path.join(path, 'logs', version, 'train_epoch_bleu.txt')) as f:
	lines = f.readlines()

train_bleu = []
for line in lines:
	train_bleu.append(float(line))

with open(os.path.join(path, 'logs', version, 'val_epoch_bleu.txt')) as f:
	lines = f.readlines()

val_bleu = []
for line in lines:
	val_bleu.append(float(line))

epoch = [i for i in range(0,5*len(train_bleu),5)]

plt.plot(epoch, train_bleu)
plt.plot(epoch, val_bleu)
plt.legend(['Train BLEU', 'Validation BLEU'])
plt.ylabel('BLEU-1')
plt.xlabel('Epoch')
plt.show()