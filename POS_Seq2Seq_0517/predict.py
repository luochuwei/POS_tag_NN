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
import cPickle
import gc
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from Seq2Seq import *
import get_data

e = 0.1   #error
lr = 0.001
drop_rate = 0.
batch_size = 40
read_data_batch = 1600
full_data_len = 142854



if full_data_len % read_data_batch == 0:
    batch_split = full_data_len/read_data_batch
else:
    batch_split = full_data_len/read_data_batch + 1


hidden_size_encoder = [1000, 1000, 1000, 1000]
hidden_size_decoder = [1000, 1000, 1000]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "sgd" 

x_path = r"data/STC/post-0517.txt"
x_tag_path = "data/STC/post-tag-0517.txt"
y_path = "data/STC/response-0517.txt"
y_tag_path = "data/STC/response-tag-0517.txt"
# test_path = "data/toy2.txt"
# test_path = "data/test-100.post"
i2w_path = 'data/STC/i2w40000_0419.pkl'
w2i_path = 'data/STC/w2i40000_0419.pkl'
i2t_path = 'data/STC/i2t_0517.pkl'
t2i_path = 'data/STC/t2i_0517.pkl'



i2w, w2i = load_data_dic(i2w_path, w2i_path)
i2t, t2i = load_data_dic(i2t_path, t2i_path)


# test_data_x_y = get_data.test_processing_long(test_path, i2w, w2i, batch_size)

print "#dic = " + str(len(w2i))
# print "unknown = " + str(tf["<UNknown>"])

dim_x = len(w2i)
dim_y = len(w2i)
dim_tag = len(t2i)
num_sents = batch_size

print "#features = ", dim_x, "#labels = ", dim_y
print "#tag len = ", dim_tag


print "load test data..."
test_batch = 15
test_data_x_y = get_data.test_processing_long(r"data/STC/validation_file-0517.txt", r"data/STC/validation_file_only_tag-0517.txt", i2w, w2i, i2t, t2i, 100, test_batch)
reference_dic = cPickle.load(open(r'print_bleu_score/reference_dic.pkl', 'rb'))
print "done."

print "compiling..."
model = Seq2Seq(dim_x + dim_tag, dim_y + dim_tag, dim_y, dim_tag, hidden_size_encoder, hidden_size_decoder, cell, optimizer, drop_rate, num_sents)
load_model('model/12_0526.model', model)

print 'loading...'
load_model("model/0525.model", model)
print 'model done'


print "predicting..."

t_bleu = []
for tlen in xrange(len(test_data_x_y)):
    p_sents_y, p_sents_t = model.predict(test_data_x_y[tlen][0], test_data_x_y[tlen][1],test_data_x_y[tlen][4], test_batch)
    get_data.print_sentence(p_sents_y, dim_y, i2w)
    candidate_dic = get_data.get_candidate_dic_for_test_pos(p_sents_y, dim_y, i2w)
    batch_bleu, _ = print_bleu_normal_batch(candidate_dic, reference_dic, tlen * test_batch)
    t_bleu.append(batch_bleu)
print "~~~~~~~~~~~~~Test Bleu is ", float(sum(t_bleu))/len(t_bleu), "~~~~~~~~~~~~~~~~"

# t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], 100)

# test_data_x_y = get_data.test_sentence_input_processing_long("a b c d", i2w, w2i, 5, 1)

# sents_y, sents_t = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][4], 2)





# get_data.print_sentence(sents_y, dim_y, i2w)
# get_data.print_sentence(sents_t, dim_tag, i2t)



# test_data_x_y_sen = get_data.test_sentence_input_processing_long("a b c", "1 2 3", i2w, w2i, i2t, t2i, 100, 1)


# sents_y, sents_t = model.predict(test_data_x_y_sen[0][0], test_data_x_y_sen[0][1],test_data_x_y_sen[0][4], 1)
# get_data.print_sentence(sents_y, dim_y, i2w)
# get_data.print_sentence(sents_t, dim_tag, i2t)




# candidate_dic = get_data.get_candidate_dic_for_test_pos(sents_y, dim_y, i2w)





# def response(sentence_seg, model, i2w, w2i):
#     test_data_x_y = get_data.test_sentence_input_processing_long(sentence_seg, i2w, w2i, 100, 1)
#     t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], 1)
#     get_data.print_sentence(t_sents[0], dim_y, i2w)
