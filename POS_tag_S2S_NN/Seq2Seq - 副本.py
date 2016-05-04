#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: Focus of attention encoder decoder network
#
#######################################################
import numpy as np
import theano
import theano.tensor as T

from gru import *
from lstm import *
from word_encoder import *
from sent_encoder import *

from word_decoder_POS import *
from updates import *

class Seq2Seq(object):
    def __init__(self, in_size, out_size, dim_y, dim_pos, hidden_size_encoder, hidden_size_decoder, cell = "gru", optimizer = "rmsprop", p = 0.5, num_sents = 1):

        self.X = T.matrix("X")
        self.Y = T.matrix("Y")
        self.in_size = in_size
        self.out_size = out_size
        self.dim_y = dim_y
        self.dim_pos = dim_pos
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.cell = cell
        self.drop_rate = p
        self.num_sents = num_sents
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.mask = T.matrix("mask")
        self.mask_y = T.matrix("mask_y")
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
                
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)
        # LM layers
        word_encoder_layer = WordEncoderLayer(rng, self.X, self.in_size, self.out_size, self.hidden_size_encoder,
                         self.cell, self.optimizer, self.drop_rate,
                         self.is_train, self.batch_size, self.mask)
        self.layers += word_encoder_layer.layers
        self.params += word_encoder_layer.params
        self.t = word_encoder_layer.hh

        i = len(self.layers) - 1

        # encoder layer
        layer_input = word_encoder_layer.activation
        self.test = word_encoder_layer.activation
        encoder_layer = SentEncoderLayer(self.cell, rng, str(i + 1), (word_encoder_layer.hidden_size, word_encoder_layer.hidden_size),
                                         layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(encoder_layer)
        
        sents_codes = encoder_layer.sent_encs
        self.test2 = sents_codes
        sents_codes = T.reshape(sents_codes, (1, self.batch_size * encoder_layer.out_size))
        self.test7 = sents_codes

        # word decoder
        word_decoder_layer = WordDecoderLayer_pos(self.cell, rng, str(i + 2), (encoder_layer.out_size, self.out_size), self.dim_y, self.dim_pos, sents_codes, self.mask_y, self.hidden_size_decoder, self.is_train, self.batch_size, self.drop_rate)

        self.layers.append(word_decoder_layer)
        self.params += word_decoder_layer.params

        self.activation = word_decoder_layer.activation
        # self.test3 = word_decoder_layer.out_size
        print word_decoder_layer.out_size


        self.predict = theano.function(inputs = [self.X, self.mask, self.mask_y, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [self.activation],on_unused_input='ignore')


    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15
    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        m = T.reshape(self.mask_y, (self.mask_y.shape[0] * self.batch_size, 1))
        ce = T.nnet.categorical_crossentropy(y_pred, y_true)
        ce = T.reshape(ce, (self.mask_y.shape[0] * self.batch_size, 1))
        return T.sum(ce * m) / T.sum(m)
        # return ce.mean()


    def define_train_test_funcs(self):
        pYs = T.reshape(self.activation, (self.mask_y.shape[0] * self.batch_size, self.out_size))
        # tYs =  T.reshape(self.X, (self.mask.shape[0] * self.batch_size, self.out_size))
        tYs =  T.reshape(self.Y, (self.mask_y.shape[0] * self.batch_size, self.out_size))
        cost = self.categorical_crossentropy(pYs, tYs)

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        #updates = sgd(self.params, gparams, lr)
        #updates = momentum(self.params, gparams, lr)
        #updates = rmsprop(self.params, gparams, lr)
        #updates = adagrad(self.params, gparams, lr)
        #updates = adadelta(self.params, gparams, lr)
        #updates = adam(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, self.Y, self.mask, self.mask_y, lr, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [cost, self.activation],
                                               updates = updates)
