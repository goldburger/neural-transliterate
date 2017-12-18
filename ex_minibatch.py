import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transliterate import *
from model_minibatch import ModelMiniBatch
import data


# plot learning curves using the array of loss values saved while training
def savePlot(points, file):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.5)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)


	plt.title('Loss Curve')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.savefig(file)

def writePlot(points, file):
	f = codecs.open(file, mode='w')
	for p in points:
		f.write(str(p))
		f.write('\n')
	f.close()


output_folder = "experiment_data/batch_size/"
if not os.path.exists(os.path.dirname(output_folder)):
	os.makedirs(os.path.dirname(output_folder))

file_path = 'data/en_bg.train.txt'
val_path = 'data/en_bg.val.txt'

# model parameters
hidden_size = 256

# training hyperparameters
alpha = 0.01
batch_size = [1, 10, 20, 40, 80, 160]
n_iter = 32000
#n_test = 300

# how verbose
printfreq = 500
plotfreq = 1

# STEP 1: read in and prepare training data
input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)

test_pairs = data.prepareTestData(val_path, input_lang, output_lang, reverse=True)
#test_pairs = [random.choice(test_pairs) for i in range(n_test)]

# STEP 2: define and train sequence to sequence model
edit_dist = []
time_taken = []
for bs in batch_size:
	start = time.time()
	n_epoch = n_iter/bs
	encoder = EncoderRNN(input_lang.n_chars, hidden_size)
	decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)
	model = ModelMiniBatch(encoder, decoder, input_lang, output_lang)
	loss_log = model.trainIters(pairs, n_epoch, print_every=printfreq, plot_every=plotfreq, learning_rate=alpha, batch_size=bs)

	end = time.time()
	print('time', end - start)
	time_taken.append(str(end - start))

	file = '{0}bs_{1}.txt'.format(output_folder, bs)
	plot = '{0}bs_{1}.pdf'.format(output_folder, bs)
	
	writePlot(loss_log, file)

	# STEP 4: evaluate the model on unseen validation examples
	print("Evaluate on unseen data")
	distance, outputs = model.generateTest(test_pairs)
	edit_dist.append(float(distance) / len(outputs))

writePlot(batch_size, output_folder+'bs_values.txt')
writePlot(edit_dist, output_folder+'distance_values.txt')
writePlot(time_taken, output_folder+'time.txt')
