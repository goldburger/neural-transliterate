import sys
from optparse import OptionParser
import codecs
import string
import random

if sys.version_info >= (3, 0):
    import queue as queue
else:
    import Queue as queue

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import utils
from nltk.metrics.distance import edit_distance

import data

SOS_token = 0
EOS_token = 1

class BeamNode():
    def __init__(self, parent_node, hidden, token_index, score):
        self.parent = parent_node
        self.hidden = hidden
        self.token_index = token_index
        self.score = score

        self.path_score = score
        if self.parent != None:
            self.path_score = self.parent.path_score + score



class ModelBeamDecode:

    def __init__(self, encoder, decoder, input_lang, output_lang, beam_width=1):

        self.beam_width = beam_width
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

    #################################################################################
    # TRAINING
    #################################################################################

    # helper functions to prepare data: to train, for each pair we will need an input tensor (indexes of the
    # characters in the input word) and target tensor (indexes of the
    # characters in the target word). We append the EOS token to both sequences.
    def indexesFromWord(self, lang, word):
        return [lang.char2index[char] for char in list(word)]

    def variableFromWord(self, lang, word):
        indexes = self.indexesFromWord(lang, word)
        indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1,1))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self, pair):
        input_variable = self.variableFromWord(self.input_lang, pair[0])
        output_variable = self.variableFromWord(self.output_lang, pair[1])
        return (input_variable, output_variable)


    # Train the model on one example
    def train(self, input_variable, target_variable, encoder_optimizer, decoder_optimizer, criterion, max_length=20):
        encoder_hidden = self.encoder.initHidden()
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < 0.5 else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]

        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)

                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
            
                if ni == EOS_token:
                    break
        
        # use autograd to backpropagate loss
        loss.backward()
        # update model parameters
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length


    # Full training process
    def trainIters(self, pairs, n_iters, print_every=100, plot_every=1000, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0 # reset every print_every
        plot_loss_total = 0 # reset every print_every

        # define criterion and optimization algorithm
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        training_pairs = [self.variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
        criterion = nn.NLLLoss()

        # now proceed one iteration at a time
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            # train on one example
            loss = self.train(input_variable, target_variable, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (utils.timeSince(start, float(iter) / float(n_iters) ), iter, float(iter) / float(n_iters) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / float(plot_every)
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # plot the learning curve
        #utils.showPlot(plot_losses)
        return plot_losses



    #################################################################################
    # GENERATE TRANSLISTERATIONS
    #################################################################################

    # Given an encoder-decoder model, and an input word, generate its transliteration
    def generate(self, word, max_length=20):
        # Create input variable and initialize 
        input_variable = self.variableFromWord(self.input_lang, word)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        # encode input word
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        
        # initialize decoder
        decoder_hidden = encoder_hidden
        # store produced characters and attention weights
        decoder_attentions = torch.zeros(max_length, max_length)

        # generate output word one character at a time
        root = BeamNode(None, decoder_hidden, SOS_token, 0)
        k = self.beam_width
        k_best = [root]
        new_hypotheses = True
        #k_best = queue.PriorityQueue()

        for di in range(max_length):
            if not new_hypotheses:
                break

            hypotheses = queue.PriorityQueue()
            new_hypotheses = False

            for node in k_best:
                if node.token_index == EOS_token:
                    break

                decoder_input = Variable(torch.LongTensor([[node.token_index]]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data

                # pick character with highest score at the output layer
                topv, topi = decoder_output.data.topk(k)

                for ki in range(k):
                    token_index = topi[0][ki]
                    token_score = topv[0][ki]
                    new_node = BeamNode(node, decoder_hidden, token_index, token_score)
                    hypotheses.put( (-new_node.path_score, new_node) )
                    new_hypotheses = True

            if not new_hypotheses:
                break

            k_best = []
            for i in range(k):
                k_best.append(hypotheses.get()[1])

        decoded_chars = []
        node = k_best[0]
        while node != None:
            char = self.output_lang.index2char[node.token_index]
            decoded_chars.append(char)
            node = node.parent
        decoded_chars.pop()
        decoded_chars.reverse()
        decoded_chars.pop()

        return decoded_chars, decoder_attentions[:di + 1]

    # Generate outputs for n randomly selected training examples
    def generateRandomly(self, n=5):
        for i in range(n) :
            pair = random.choice(pairs)
            print('INPUT: ',pair[0])
            print('TARGET: ',pair[1])
            output_chars, attentions = self.generate(pair[0])
            print ('OUTPUT: ', ''.join(output_chars))
            print('')


    # Generate outputs for the given sequence of test pairs, and compute
    # the total edit distance between system output and target transliteration
    def generateTest(self, test_pairs):
        score = 0
        outputs = []
        for pair in test_pairs:
            output_chars, attentions = self.generate(pair[0])
            score += edit_distance(''.join(output_chars).replace('<EOS>', ''),pair[1])
            outputs.append(''.join(output_chars).replace('<EOS>',''))
        return score, outputs

