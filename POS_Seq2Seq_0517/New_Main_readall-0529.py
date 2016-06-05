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
lr = 0.01
drop_rate = 0.
batch_size = 100
read_data_batch = 2100
# full_data_len = 190363

full_data_len = 2100
# hidden_size_encoder = hidden_size_decoder = [100]
hidden_size_encoder = [1000, 1000, 1000, 1000]
hidden_size_decoder = [1000, 1000, 1000]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "sgd" 






# batch_size = 10
# read_data_batch = 10
# # full_data_len = 190363

# full_data_len = 10

x_path = "data/nba2/basketball-post-0529.txt"
x_tag_path = "data/nba2/basketball-post-0529-tag.txt"
y_path = "data/nba2/basketball-response-0529.txt"
y_tag_path = "data/nba2/basketball-response-0529-tag.txt"
# test_path = "data/toy2.txt"


threshold = 0


# _, _, i2w, w2i, tf, _ = get_data.processing(x_path, y_path, threshold, 0, 1, 1)
X_seqs, y_seqs, y_tag_seqs, i2w, w2i, t2i, i2t, tf, data_x_y = get_data.processing(x_path, y_path, x_tag_path, y_tag_path, threshold, 0, 2100, batch_size)
# test_data_x_y = get_data.test_processing(test_path, i2w, w2i, batch_size)

print "#dic = " + str(len(w2i))
# print "unknown = " + str(tf["<UNknown>"])

dim_x = len(w2i)
dim_y = len(w2i)
dim_tag = len(t2i)
num_sents = batch_size


print "save data dic..."
save_data_dic("data/nba2/i2w-test-t.pkl", "data/nba2/w2i-test-t.pkl", i2w, w2i)
save_data_dic("data/nba2/i2t-test-t.pkl", "data/nba2/t2i-test-t.pkl", i2t, t2i)

print "#features = ", dim_x, "#labels = ", dim_y
print "#tag len = ", dim_tag

print "compiling..."
model = Seq2Seq(dim_x + dim_tag, dim_y + dim_tag, dim_y, dim_tag, hidden_size_encoder, hidden_size_decoder, cell, optimizer, drop_rate, num_sents)
# # load_error_model("GRU-200_best.model", model)
load_model(r"data/nba2/model/0531.model", model)

print "training..."


start = time.time()
g_error = 0.12
for i in xrange(50000):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_x_y.items():
        X = xy[0]
        mask = xy[1]
        Y = xy[2]
        Yt = xy[3]
        mask_y = xy[4]
        local_batch_size = xy[5]

        cost, cost_y, cost_pos, sents_y, sents_t = model.train(X, Y, Yt, mask, mask_y, lr, local_batch_size)
        
        error += cost
        # break
        
    in_b_time = time.time() - in_start
    # break

        # l,r = model.predict(data_t1[0][0], data_t1[0][1],data_t1[0][3], data_t1[0][5], 1)
        # l2,r2 = model.predict(data_t2[0][0], data_t2[0][1],data_t2[0][3], data_t2[0][5], 1)
        # l3,r3 = model.predict(data_4[0][0], data_4[0][1],data_4[0][3], data_4[0][5], 1)
        # t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], batch_size)
        #打印结果

        # print "Test : "
        # get_data.print_sentence(l, dim_y, i2w)
        # get_data.print_sentence(l2, dim_y, i2w)
        # get_data.print_sentence_last_n(t_sents[0], dim_y, i2w, 5)

    error /= len(data_x_y);
    print "Iter = " + str(i)+ " Error = " + str(error) + ", Time = " + str(in_b_time)
    get_data.print_sentence_last_n(sents_y, dim_y, i2w, 4)
    get_data.print_sentence_last_n(sents_t, dim_tag, i2t, 4)

    if error < g_error:
    	print "smaller error, save model..."
    	save_model('data/nba2/model/0602.model', model)
    	g_error = error

    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("data/nba2/model/0602-new.model", model)
