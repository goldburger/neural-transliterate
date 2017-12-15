import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transliterate import *
from model import Model
from model_noattention import ModelNoAttention
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


output_folder = "experiment_data/attention/"
if not os.path.exists(os.path.dirname(output_folder)):
	os.makedirs(os.path.dirname(output_folder))

file_path = 'data/en_bg.train.txt'
val_path = 'data/en_bg.val.txt'

# model parameters
hidden_size = 256

# training hyperparameters
learn_rate = 0.01
n_iter = 40000
#n_test = 300

# how verbose
printfreq = 1000
plotfreq = 1

# STEP 1: read in and prepare training data
input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)

test_pairs = data.prepareTestData(val_path, input_lang, output_lang, reverse=True)
#test_pairs = [random.choice(test_pairs) for i in range(n_test)]

# STEP 2: define and train sequence to sequence model
encoder1 = EncoderRNN(input_lang.n_chars, hidden_size)
decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)

encoder2 = EncoderRNN(input_lang.n_chars, hidden_size)
decoder2 = DecoderRNN(hidden_size, output_lang.n_chars, 1)

model1 = Model(encoder1, decoder1, input_lang, output_lang)
model2 = ModelNoAttention(encoder2, decoder2, input_lang, output_lang)

edit_dist1 = []
edit_dist2 = []
loss_log1 = []
loss_log2 = []
iter_log = []

for i in range(1, int(n_iter/1000)+1):
	print('Training model 1')
	loss1 = model1.trainIters(pairs, 1000, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)
	print('Training model 2')
	loss2 = model2.trainIters(pairs, 1000, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)

	loss_log1 += loss1
	loss_log2 += loss2

	# STEP 4: evaluate the model on unseen validation examples
	print("Evaluate model1 on unseen data")
	distance, outputs = model1.generateTest(test_pairs)
	edit_dist1.append(float(distance) / len(outputs))

	print("Evaluate model2 on unseen data")
	distance, outputs = model2.generateTest(test_pairs)
	edit_dist2.append(float(distance) / len(outputs))

	iter_log.append(i*1000)

writePlot(loss_log1, output_folder+'attn_loss_curve.txt')
writePlot(loss_log2, output_folder+'no_attn_loss_curve.txt')
writePlot(edit_dist1, output_folder+'attn_dist_curve.txt')
writePlot(edit_dist2, output_folder+'no_attn_dist_curve.txt')
writePlot(iter_log, output_folder+'iter_values.txt')
