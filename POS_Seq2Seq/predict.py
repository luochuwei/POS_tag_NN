#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 21/02/2016
#    Usage: new Main (in case of the out of memory)
#
####################################################

import time
import gc
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from Seq2Seq import *
import get_data


e = 0.01   #error
lr = 1
drop_rate = 0.
batch_size = 3
read_data_batch = 3
# full_data_len = 190363

full_data_len = 3
hidden_size_encoder = [2,3]
hidden_size_decoder = [4,5]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 


print "loading dic..."
i2w, w2i = load_data_dic(r'data/i2w-test.pkl', r'data/w2i-test.pkl')
print "done"

print "#dic = " + str(len(w2i))

dim_x = len(w2i)
dim_y = len(w2i)
num_sents = batch_size

print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = Seq2Seq(dim_x, dim_y, hidden_size_encoder, hidden_size_decoder, cell, optimizer, drop_rate, num_sents)

print 'loading...'
load_model("0420-new.model", model)
print 'model done'


print "predicting..."


# test_data_x_y = get_data.test_processing_long(r'data/SMT-test-100.post', i2w, w2i, 100, 100)



# t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], 100)

test_data_x_y = get_data.test_sentence_input_processing_long("a b c d", i2w, w2i, 5, 1)

t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], 1)





get_data.print_sentence(t_sents[0], dim_y, i2w)

def response(sentence_seg, model, i2w, w2i):
    test_data_x_y = get_data.test_sentence_input_processing_long(sentence_seg, i2w, w2i, 100, 1)
    t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], 1)
    get_data.print_sentence(t_sents[0], dim_y, i2w)
