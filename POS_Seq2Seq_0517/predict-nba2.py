#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 02/06/2016
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
from print_bleu_score.bleu_print import *


e = 0.1   #error
lr = 0.001
drop_rate = 0.
batch_size = 100
read_data_batch = 100
full_data_len = 100



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

# x_path = r"data/STC/post-0517.txt"
# x_tag_path = "data/STC/post-tag-0517.txt"
# y_path = "data/STC/response-0517.txt"
# y_tag_path = "data/STC/response-tag-0517.txt"
# test_path = "data/toy2.txt"
# test_path = "data/test-100.post"
i2w_path = 'data/nba2/i2w-test-t.pkl'
w2i_path = 'data/nba2/w2i-test-t.pkl'
i2t_path = 'data/nba2/i2t-test-t.pkl'
t2i_path = 'data/nba2/t2i-test-t.pkl'



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
test_batch = 170
test_data_x_y = get_data.test_processing_long(r"evaluation/testfile-post.txt", r"evaluation/testfile-only-tag.txt", i2w, w2i, i2t, t2i, 100, test_batch)
reference_dic = cPickle.load(open(r'data/nba2/reference_dic_for_nba_2-500.pkl', 'rb'))
print "done."




print "compiling..."
model = Seq2Seq(dim_x + dim_tag, dim_y + dim_tag, dim_y, dim_tag, hidden_size_encoder, hidden_size_decoder, cell, optimizer, drop_rate, num_sents)


print 'loading model...'
load_model(r'data/nba2/model/0531 - 0.122.model', model)
print 'model done'

p_sents_y, p_sents_t = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][4], test_batch)
get_data.print_sentence(p_sents_y, dim_y, i2w)


# print "predicting..."

# t_bleu = []
# for tlen in xrange(len(test_data_x_y)):
#     p_sents_y, p_sents_t = model.predict(test_data_x_y[tlen][0], test_data_x_y[tlen][1],test_data_x_y[tlen][4], test_batch)
#     get_data.print_sentence(p_sents_y, dim_y, i2w)
#     candidate_dic = get_data.get_candidate_dic_for_test_pos(p_sents_y, dim_y, i2w)
#     batch_bleu, _ = print_bleu_normal_batch(candidate_dic, reference_dic, tlen * test_batch)
#     t_bleu.append(batch_bleu)
# print "~~~~~~~~~~~~~Test Bleu is ", float(sum(t_bleu))/len(t_bleu), "~~~~~~~~~~~~~~~~"

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
