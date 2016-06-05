#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 16/05/2016
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
from print_bleu_score.bleu_print import *

e = 0.1   #error
lr = 0.008
drop_rate = 0.
batch_size = 100
read_data_batch = 2000
full_data_len = 4000



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

x_path = "data/nba/basketball-post_0527.txt"
x_tag_path = "data/nba/basketball-post_tag_0527.txt"
y_path = "data/nba/basketball-response_0527.txt"
y_tag_path = "data/nba/basketball-response-tag_0527.txt"
# test_path = "data/toy2.txt"
# test_path = "data/test-100.post"

threshold = 0


# _, _, i2w, w2i, tf, _ = get_data.processing(x_path, y_path, threshold, 0, 1, 1)
X_seqs, y_seqs, y_tag_seqs, i2w, w2i, t2i, i2t, tf, data_x_y = get_data.processing(x_path, y_path, x_tag_path, y_tag_path, threshold, 0, 4000, batch_size)
# test_data_x_y = get_data.test_processing(test_path, i2w, w2i, batch_size)

print "#dic = " + str(len(w2i))
# print "unknown = " + str(tf["<UNknown>"])

dim_x = len(w2i)
dim_y = len(w2i)
dim_tag = len(t2i)
num_sents = batch_size


print "save data dic..."
save_data_dic("data/nba/i2w-test-t.pkl", "data/nba/w2i-test-t.pkl", i2w, w2i)
save_data_dic("data/nba/i2t-test-t.pkl", "data/nba/t2i-test-t.pkl", i2t, t2i)

print "#features = ", dim_x, "#labels = ", dim_y
print "#tag len = ", dim_tag

print "compiling..."
model = Seq2Seq(dim_x + dim_tag, dim_y + dim_tag, dim_y, dim_tag, hidden_size_encoder, hidden_size_decoder, cell, optimizer, drop_rate, num_sents)
# # load_error_model("GRU-200_best.model", model)
load_model(r'model/nba/full-87-iter.model', model)

print "training..."



start = time.time()
g_error = 1
for i in xrange(10000):
    error = 0.0
    in_start = time.time()
    for get_num_start in xrange((full_data_len/read_data_batch)+1):
        read_data_batch_error = 0.0
        in_b_start = time.time()
        get_num_end = get_num_start*read_data_batch + read_data_batch
        if get_num_end > full_data_len:
            x = full_data_len - get_num_start*read_data_batch 
            # get_num_end = get_num_start*read_data_batch + x/batch_size*batch_size
            get_num_end = full_data_len
        if get_num_start*read_data_batch == full_data_len:
            break
        # _, _, _, _, data_x_y = get_data.new_processing(x_path, y_path, w2i, i2w, get_num_start*read_data_batch, get_num_end, batch_size)
        _, _, _, _, _, _, _, data_x_y = get_data.new_processing(x_path, y_path, x_tag_path, y_tag_path, i2w, w2i, i2t, t2i, get_num_start*read_data_batch, get_num_end, batch_size)
        for batch_id, xy in data_x_y.items():
            X = xy[0]
            mask = xy[1]
            Y = xy[2]
            Yt = xy[3]
            mask_y = xy[4]
            local_batch_size = xy[5]

            cost, cy, cp, sents_y, sents_t = model.train(X, Y, Yt, mask, mask_y, lr, local_batch_size)
            #print "cy is ", cy
            #print "cp is ", cp
            #print "gp is ", gp
            read_data_batch_error += cost
        
        in_b_time = time.time() - in_b_start
        # break

        

        read_data_batch_error /= len(data_x_y);
        error += read_data_batch_error

        del data_x_y
        gc.collect()
        print "Minibatch_Iter = " + str(get_num_start)+ ", "+ str(100*(get_num_start+1)/float(batch_split)) + "%, read_batch_Error = " + str(read_data_batch_error) + ", Time = " + str(in_b_time)

        if read_data_batch_error < g_error:
            g_error = read_data_batch_error
            print 'new smaller cost, save param...'
            save_model(r"model/nba/full_test.model", model)

            # p_sents_y, p_sents_t = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][4], test_batch)
            # print ".........................predict....................."
            # get_data.print_sentence(p_sents_y, dim_y, i2w)
            # get_data.print_sentence(p_sents_t, dim_y, i2t)
            # print ".....................predict done...................."
            # ^ predict example ^
        # t_bleu = []
        # for tlen in xrange(len(test_data_x_y)):
        #     p_sents_y, p_sents_t = model.predict(test_data_x_y[tlen][0], test_data_x_y[tlen][1],test_data_x_y[tlen][4], test_batch)
        #     candidate_dic = get_data.get_candidate_dic_for_test_pos(p_sents_y, dim_y, i2w)
        #     batch_bleu, _ = print_bleu_normal_batch(candidate_dic, reference_dic, tlen * test_batch)
        #     t_bleu.append(batch_bleu)
        # print "~~~~~~~~~~~~~Test Bleu is ", float(sum(t_bleu))/len(t_bleu), "~~~~~~~~~~~~~~~~"
        # if float(sum(t_bleu))/len(t_bleu) > 0.7:
        #     save_model(r"model/above_07.model", model)
        get_data.print_sentence_last_n(sents_y, dim_y, i2w, 4)
        get_data.print_sentence_last_n(sents_t, dim_tag, i2t, 4)

        
        print "read batch train last :"
        get_data.print_sentence_last_n(sents_y, dim_y, i2w, 4)
        get_data.print_sentence_last_n(sents_t, dim_tag, i2t, 4)


    E = float(error)/batch_split

    # if E < g_error:
    #     g_error = E
    #     print 'new smaller cost, save param...'
    #     save_model("0405-GRU-all-4hidden1000_best.model", model)
    # if error < 3.0:
    #     t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], test_data_x_y[0][-1])
        #get_data.print_sentence(t_sents[0], dim_y, i2w)     
    print i, 'iter', "cost is ", E
    print "save model..."
    save_model(r"model/nba/full-"+str(i)+"-iter-0528.model", model)
    # print "train_last :"
    # get_data.print_sentence(sents, dim_y, i2w, 5)
    if E <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model(r"model/nba/final.model", model)
