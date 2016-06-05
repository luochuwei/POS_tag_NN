#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 20/04/2016
#    Usage: decoder
#
#######################################################
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from lstm import *
from gru import *

class WordDecoderLayer_pos(object):
    def __init__(self, cell, rng, layer_id, shape, dim_y, dim_pos, X, mask, hidden_size, is_train = 1, batch_size = 1, p = 0.5):
        self.prefix = "WordDecoderLayer_"
        self.layer_id = "_" + layer_id
        self.out_size, self.in_size = shape
        print "111111"
        print self.out_size
        print self.in_size
        print "111111"
        self.dim_y = dim_y
        self.dim_pos = dim_pos
        if self.in_size == self.dim_y + self.dim_pos:
            print "decoder size right !"
        # assert self.dim_pos + self.dim_y == self.in_size
        self.mask = mask
        self.X = X
        self.words = mask.shape[0]
        self.hidden_size_list = hidden_size
        self.num_hds = len(hidden_size)
        self.cell = cell
        self.rng = rng
        self.is_train = is_train
        self.batch_size = batch_size
        self.drop_rate = p


        self.define_layers()
        
        
        

    

    def define_layers(self):
        self.layers = []
        self.params = []

        for i in xrange(self.num_hds):
            if i == 0:
                layer_input = self.X
                h_shape = (self.out_size, self.hidden_size_list[0])
            else:
                layer_input = self.layers[i - 1].activation
                h_shape = (self.hidden_size_list[i - 1], self.hidden_size_list[i])

            if self.cell == "gru":
                hidden_layer = GRULayer(self.rng, self.prefix + self.layer_id + str(i), h_shape, layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
            elif self.cell == "lstm":
                hidden_layer = LSTMLayer(self.rng, self.prefix + self.layer_id + str(i), h_shape, layer_input,
                                         self.mask, self.is_train, self.batch_size, self.drop_rate)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params


        #the last decoder layer for decoding
        if self.num_hds == 0:
            output_layer_input = self.X
            last_shape = (self.in_size, self.out_size)
        else:
            output_layer_input = self.layers[-1].activation
            last_shape = (self.in_size, self.layers[-1].out_size)
        
        self.W_hy = init_weights((last_shape[1], last_shape[0]), self.prefix + "W_hy" + self.layer_id)
        self.b_y = init_bias(last_shape[0], self.prefix + "b_y" + self.layer_id)


        self.word_tag_matrix = init_weights((self.dim_pos, self.dim_y), "word_tag_matrix")
        print "~~~~"
        print last_shape[0]
        print last_shape[1]
        print "~~~~"
        if self.cell == "gru":
            self.decoder = GRULayer(self.rng, self.prefix + self.layer_id, last_shape, output_layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
            def _active(m, pre_h, x):
                x = T.reshape(x, (self.batch_size, last_shape[0]))

                pre_h = T.reshape(pre_h, (self.batch_size, last_shape[1]))

                h = self.decoder._active(x, pre_h)

                # y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
                y = T.dot(h, self.W_hy) + self.b_y
                # y = y * m[:, None]


                y_dim_y = y[:, 0:self.dim_y]
                y_dim_pos = y[:, self.dim_y:]

                new_y_dim_y = T.nnet.softmax(y_dim_y + T.dot(y_dim_pos, self.word_tag_matrix))
                y_dim_pos = T.nnet.softmax(y_dim_pos)

                # y = np.column_stack((new_y_dim_y, y_dim_pos))
                y = T.concatenate([new_y_dim_y, y_dim_pos], axis=1)

                y = y * m[:, None]
                new_y_dim_y = new_y_dim_y * m[:, None]
                y_dim_pos = y_dim_pos * m[:, None]

                new_y_dim_y = T.reshape(new_y_dim_y, (1, self.batch_size * self.dim_y))
                y_dim_pos = T.reshape(y_dim_pos, (1, self.batch_size * self.dim_pos))
                h = T.reshape(h, (1, self.batch_size * last_shape[1]))
                y = T.reshape(y, (1, self.batch_size * last_shape[0]))
                return h, y, new_y_dim_y, y_dim_pos
                # return h, y
            [h, y, new_y_dim_y, y_dim_pos], updates = theano.scan(_active, #n_steps = self.words,
                                      sequences = [self.mask],
                                      outputs_info = [{'initial':output_layer_input, 'taps':[-1]},
                                      T.alloc(floatX(0.), 1, self.batch_size * last_shape[0]), None, None])
        elif self.cell == "lstm":
            self.decoder = LSTMLayer(self.rng, self.prefix + self.layer_id, last_shape, output_layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
            def _active(m, pre_h, pre_c, x):
                x = T.reshape(x, (self.batch_size, last_shape[0]))
                pre_h = T.reshape(pre_h, (self.batch_size, last_shape[1]))
                pre_c = T.reshape(pre_c, (self.batch_size, last_shape[1]))

                h, c = self.decoder._active(x, pre_h, pre_c)
            
                y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
                y = y * m[:, None]

                h = T.reshape(h, (1, self.batch_size * last_shape[1]))
                c = T.reshape(c, (1, self.batch_size * last_shape[1]))
                y = T.reshape(y, (1, self.batch_size * last_shape[0]))
                return h, c, y
            [h, c, y], updates = theano.scan(_active, sequences = [self.mask],outputs_info = [{'initial':output_layer_input, 'taps':[-1]}, {'initial':output_layer_input, 'taps':[-1]}, T.alloc(floatX(0.), 1, self.batch_size * last_shape[0])])
        
        y = T.reshape(y, (self.words, self.batch_size * last_shape[0]))
        self.activation = y
        new_y_dim_y = T.reshape(new_y_dim_y, (self.words, self.batch_size * self.dim_y))
        self.activation_dim_y = new_y_dim_y
        y_dim_pos = T.reshape(y_dim_pos, (self.words, self.batch_size * self.dim_pos))
        self.activation_dim_pos = y_dim_pos
        self.params += self.decoder.params
        self.params += [self.W_hy, self.b_y]
        self.params += [self.word_tag_matrix]
        # self.layers.append(self.decoder)
        # self.hhhh = h

