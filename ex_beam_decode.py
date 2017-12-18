import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transliterate import *
from model import Model
from model_beamdecode import ModelBeamDecode
import data

def writePlot(points, file):
	f = codecs.open(file, mode='w')
	for p in points:
		f.write(str(p))
		f.write('\n')
	f.close()


output_folder = "experiment_data/beam_width/"
if not os.path.exists(os.path.dirname(output_folder)):
	os.makedirs(os.path.dirname(output_folder))

file_path = 'data/en_bg.train.txt'
val_path = 'data/en_bg.val.txt'

# model parameters
hidden_size = 256

# training hyperparameters
learn_rate = 0.005
n_iter = 20000
#n_test = 300

# how verbose
printfreq = 1000
plotfreq = 1

ks = [1,2,4,8]

# STEP 1: read in and prepare training data
input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)

test_pairs = data.prepareTestData(val_path, input_lang, output_lang, reverse=True)
#test_pairs = [random.choice(test_pairs) for i in range(n_test)]

# STEP 2: define and train sequence to sequence model
encoder = EncoderRNN(input_lang.n_chars, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)

model1 = Model(encoder, decoder, input_lang, output_lang)
model2 = ModelBeamDecode(encoder, decoder, input_lang, output_lang)

edit_dist1 = []
edit_dist_k = [[],[],[],[]]
loss_log = []
iter_log = []

res = 1000

for i in range(1, int(n_iter/res)+1):
	print('epoch '+ str(i))
	loss = model1.trainIters(pairs, res, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)
	loss_log += loss

	# STEP 4: evaluate the model on unseen validation examples
	print("Evaluate model1 on unseen data")
	distance, outputs = model1.generateTest(test_pairs)
	edit_dist1.append(float(distance) / len(outputs))

	for j in range(len(ks)):
		k = ks[j]
		print("Evaluate model2 on unseen data")
		model2.beam_width = k
		distance, outputs = model2.generateTest(test_pairs)
		edit_dist_k[j].append(float(distance) / len(outputs))

	iter_log.append(i*res)

writePlot(loss_log, output_folder+'loss_curve.txt')
writePlot(edit_dist1, output_folder+'k_0.txt')
writePlot(iter_log, output_folder+'iter_values.txt')

for j in range(len(ks)):
	file_name = '{0}k_{1}.txt'.format(output_folder, str(ks[j]))
	file_outputs = '{0}k_{1}.out'.format(output_folder, str(ks[j]))
	writePlot(edit_dist_k[j], file_name)

	## Example outputs
	model2.beam_width = ks[j]
	distance, outputs = model2.generateTest(test_pairs)
	if len(outputs) > 0:
		f = codecs.open(file_outputs, mode='w', encoding='utf-8')
		for o in outputs:
			f.write(o)
			f.write('\n')
		f.close()

## Example outputs
distance, outputs = model1.generateTest(test_pairs)
if len(outputs) > 0:
	f = codecs.open(output_folder+'k_0.out', mode='w', encoding='utf-8')
	for o in outputs:
		f.write(o)
		f.write('\n')
	f.close()