import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transliterate import *
from model import Model
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


output_folder = "experiment_data/num_iter/"
if not os.path.exists(os.path.dirname(output_folder)):
	os.makedirs(os.path.dirname(output_folder))

file_path = 'data/en_bg.val.txt'
val_path = 'data/en_bg.train.txt'

# model parameters
hidden_size = 256

# training hyperparameters
learn_rate = 0.01
n_iter = 500
n_test = 100

# how verbose
printfreq = 1000
plotfreq = 1

# STEP 1: read in and prepare training data
input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)

test_pairs = data.prepareTestData(val_path, input_lang, output_lang, reverse=True)
test_pairs = [random.choice(test_pairs) for i in range(n_test)]

# STEP 2: define and train sequence to sequence model
encoder = EncoderRNN(input_lang.n_chars, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)

model = Model(encoder, decoder, input_lang, output_lang)
edit_dist = []
loss_log = []
iter_log = []

for i in range(1, int(n_iter/100)+1):
	loss = model.trainIters(pairs, 100, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)
	loss_log += loss

	# STEP 4: evaluate the model on unseen validation examples
	#print("Evaluate on unseen data")
	distance, outputs = model.generateTest(test_pairs)
	edit_dist.append(float(distance) / len(outputs))
	iter_log.append(i*100)

writePlot(loss_log, output_folder+'loss_curve.txt')
writePlot(iter_log, output_folder+'iter_values.txt')
writePlot(edit_dist, output_folder+'distance_values.txt')
savePlot(edit_dist, output_folder+'distance_curve.pdf')
