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


output_folder = "experiment_data/learning_rate/"
if not os.path.exists(os.path.dirname(output_folder)):
	os.makedirs(os.path.dirname(output_folder))

file_path = 'data/en_bg.train.txt'
val_path = 'data/en_bg.val.txt'
out_path = 'out.txt'

# model parameters
hidden_size = 256

# training hyperparameters
learn_rate = [0.001, 0.005, 0.01, 0.02, 0.04]
n_iter = 30000
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
for alpha in learn_rate:
	encoder = EncoderRNN(input_lang.n_chars, hidden_size)
	decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)
	model = Model(encoder, decoder, input_lang, output_lang)
	loss_log = model.trainIters(pairs, n_iter, print_every=printfreq, plot_every=plotfreq, learning_rate=alpha)

	file = '{0}alpha_{1}.txt'.format(output_folder, alpha)
	plot = '{0}alpha_{1}.pdf'.format(output_folder, alpha)

	try:
		writePlot(loss_log, file)
		savePlot(loss_log, plot)
	except:
		pass

	# STEP 4: evaluate the model on unseen validation examples
	print("Evaluate on unseen data")
	distance, outputs = model.generateTest(test_pairs)
	edit_dist.append(float(distance) / len(outputs))

writePlot(learn_rate, output_folder+'alpha_values.txt')
writePlot(edit_dist, output_folder+'distance_values.txt')
