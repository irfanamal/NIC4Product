import matplotlib.pyplot as plt
import os
import sys

path = sys.path[0]
version = input()

with open(os.path.join(path, 'logs', version, 'train_epoch_loss.txt')) as f:
	lines = f.readlines()

train_loss = []
for line in lines:
	train_loss.append(float(line))

with open(os.path.join(path, 'logs', version, 'val_epoch_loss.txt')) as f:
	lines = f.readlines()

val_loss = []
for line in lines:
	val_loss.append(float(line))

epoch = [i for i in range(len(train_loss))]

plt.plot(epoch, train_loss)
plt.plot(epoch, val_loss)
plt.legend(['Train Loss', 'Validation Loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()