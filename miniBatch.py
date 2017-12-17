import sys
from optparse import OptionParser
import codecs
import string
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import utils
from nltk.metrics.distance import edit_distance

import data


use_cuda = torch.cuda.is_available()


SOS_token = 0   # start of sequence token
EOS_token = 1   # end of sequence token

MAX_LENGTH = 20
BATCH_SIZE = 5


#################################################################################
# DEFINE ENCODER DECODER MODELS
#################################################################################

################
# An Encoder model
# a visualization of the encoer architecture can be found here:
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder
# the RNN cell is a Gated Recurrent Unit (GRU), which adds gates to a standard RNN cell to 
# avoid vanishing/exploding gradients. 
################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    # Note that we only have to define the forward function. It 
    # will be used to construct the computation graph dynamically
    # for each new example. The backward function is automatically
    # defined for us by autograd.
    def forward(self, input, hidden):
        embedded = self.embedding(input) #.view(1, 1, -1)

        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

################
# Decoder models
################


# RNN with attention and dropout
# as illustrated here http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):

        embedded = self.embedding(input) #.view(1, 1, -1)
        embedded = self.dropout(embedded)


        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 2)), dim=1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)        
        output = self.attn_combine(output)

        for i in range(self.n_layers):
            output = F.relu(output) 
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output), dim=2)

        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        

#################################################################################
# TRAINING
#################################################################################

# helper functions to prepare data: to train, for each pair we will need an input tensor (indexes of the
# characters in the input word) and target tensor (indexes of the
# characters in the target word). We append the EOS token to both sequences.
def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]

def variableFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    while len(indexes) < MAX_LENGTH:
        indexes.append(EOS_token)

    result = Variable(torch.LongTensor(indexes).view(-1,1))

    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, input_lang, output_lang):
    input_variable = variableFromWord(input_lang, pair[0])
    output_variable = variableFromWord(output_lang, pair[1])
    return (input_variable, output_variable)


## Train the model on a batch of examples
def train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    batch_size = input_batch.size()[0]

    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(max_length):
        input_i = input_batch[:, ei]
        encoder_output, encoder_hidden = encoder(input_i, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]*batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    encoder_outputs = encoder_outputs.unsqueeze(0).repeat(batch_size, 1, 1)

    loss_mask = torch.FloatTensor([1]*batch_size)
    mask = torch.ones(batch_size, MAX_LENGTH)
    #loss = Variable(torch.zeros(batch_size, MAX_LENGTH))
    #pred = Variable(torch.zeros(batch_size, MAX_LENGTH))
    for di in range(max_length):
        decoder_hidden = decoder_hidden.repeat(batch_size,1,1)
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        
        decoder_output = decoder_output.squeeze(1)

        topv, topi = decoder_output.data.topk(1)
        ni = topi
        
        decoder_input = Variable(torch.LongTensor(ni))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        targets = target_batch[:, di]
        targets = targets.squeeze()

        loss_batch = criterion(decoder_output, targets) #.view(-1, 1)
        #loss[:, di] = loss_batch

        loss_batch = loss_batch * Variable(loss_mask)
        loss += loss_batch.sum()

        mask[:, di] = loss_mask
        loss_mask = (targets.data != EOS_token) & (loss_mask > 0)
        loss_mask = loss_mask.float()

        #pred[:, di] = topi

    #mask = Variable(mask)
    #loss = loss * mask
    #loss = loss.sum()

    # use autograd to backpropagate loss
    loss.backward()
    # update model parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / max_length


# Full training process
def trainIters(pairs, input_lang, output_lang, encoder, decoder, n_epoch, print_every=3, plot_every=3, learning_rate=0.01, batch_size=BATCH_SIZE):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # reset every print_every
    plot_loss_total = 0 # reset every print_every

    # define criterion and optimization algorithm
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(reduce=False)

    training_pairs = [variablesFromPair(random.choice(pairs), input_lang, output_lang) for i in range(n_epoch*batch_size)]
    inputs, targets = zip(*training_pairs)

    # now proceed one iteration at a time
    for epoch in range(1, n_epoch + 1):

        i_start = batch_size* (epoch-1)
        i_end = i_start+batch_size

        input_batch = torch.stack( inputs[i_start:i_end] )
        target_batch = torch.stack( targets[i_start:i_end] )

        # train on one example
        loss = train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, float(iter) / float(n_iters) ), iter, float(iter) / float(n_iters) * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / float(plot_every)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # plot the learning curve
    utils.showPlot(plot_losses)



#################################################################################
# GENERATE TRANSLISTERATIONS
#################################################################################

# Given an encoder-decoder model, and an input word, generate its transliteration
def generate(encoder, decoder, word, max_length=MAX_LENGTH):
    # Create input variable and initialize 
    input_variable = variableFromWord(input_lang, word)
    input_variable = input_variable.unsqueeze(0)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # encode input word
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    
    # initialize decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    encoder_outputs = encoder_outputs.unsqueeze(0)

    # store produced characters and attention weights
    decoded_chars = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # generate output word one character at a time
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data

        decoder_output = decoder_output.squeeze(1)

        # pick character with highest score at the output layer
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        # if the EOS token is produced, stop, otherwise go to next step
        if ni == EOS_token:
            decoded_chars.append('<EOS>')
            break
        else:
            decoded_chars.append(output_lang.index2char[ni])
            
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_chars, decoder_attentions[:di + 1]


# Generate outputs for n randomly selected training examples
def generateRandomly(encoder, decoder, n=5):
    for i in range(n) :
        pair = random.choice(pairs)
        print('INPUT: ',pair[0])
        print('TARGET: ',pair[1])
        output_chars, attentions = generate(encoder, decoder, pair[0])
        print ('OUTPUT: ', ''.join(output_chars))
        print('')


# Generate outputs for the given sequence of test pairs, and compute
# the total edit distance between system output and target transliteration
def generateTest(encoder, decoder, test_pairs):
    score = 0
    outputs = []
    for pair in test_pairs:
        output_chars, attentions = generate(encoder, decoder, pair[0])
        score += edit_distance(''.join(output_chars).replace('<EOS>', ''),pair[1])
        outputs.append(''.join(output_chars).replace('<EOS>',''))
    return score, outputs


#################################################################################
# PUT IT ALL TOGETHER
#################################################################################
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--training-set", dest="file_path",
                      help="path to training set file", metavar="FILE")
    parser.add_option("-v", "--validation-set", dest="val_path",
                      help="path to validation set file", metavar="FILE")
    parser.add_option("-o", "--output-file", dest="out_path",
                      help="transliteration output for words in validation set", metavar="FILE")
    parser.add_option("-n", "--n-iterations", dest="iterations",
                      help="number of training iterations", type='int')
    
    (options, args) = parser.parse_args()

    # model parameters
    hidden_size = 256

    # training hyperparameters
    learn_rate = 0.01
    n_iter = options.iterations
    
    # how verbose
    printfreq = 1000
    plotfreq = 1
    
    # STEP 1: read in and prepare training data
    input_lang, output_lang, pairs = data.prepareTrainData(options.file_path, 'en', 'bg', reverse=True)
    
    # STEP 2: define and train sequence to sequence model
    encoder = EncoderRNN(input_lang.n_chars, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        
    trainIters(pairs, input_lang, output_lang, encoder, decoder, n_iter, print_every=printfreq, plot_every=plotfreq, learning_rate=learn_rate)
    
    # STEP 3: generate transliteration output for a random sample of training examples
    print("Examples of output for a random sample of training examples")
    generateRandomly(encoder, decoder)
    
    # STEP 4: evaluate the model on unseen validation examples
    print("Evaluate on unseen data")
    test_pairs = data.prepareTestData(options.val_path, input_lang, output_lang, reverse=True)
    distance, outputs = generateTest(encoder, decoder, test_pairs)
    if len(outputs) > 0:
        print ("Average edit distance %.4f" % (float(distance) / len(outputs)))
        #f = codecs.open(options.out_path, mode='wt', encoding='utf-8')
        #for o in outputs:
        #    f.write(o)
        #    f.write('\n')
        #f.close()

