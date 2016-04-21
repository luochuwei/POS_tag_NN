#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: data processing
#
############################################


import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))



def print_sentence(sents, dim_y, i2w):
    for s in xrange(int(sents.shape[1] / dim_y)):
        xs = sents[:, s * dim_y : (s + 1) * dim_y]
        for w_i in xrange(xs.shape[0]):
            w = i2w[np.argmax(xs[w_i, :])]
            if w == "<eoss>":
                break
            elif w_i != 0 and w == "<soss>":
                break
            print w.decode('utf-8')," ",
        print "\n"

def print_sentence_last_n(sents, dim_y, i2w, n):
    for s in xrange(int(sents.shape[1] / dim_y)-n, int(sents.shape[1] / dim_y)):
        xs = sents[:, s * dim_y : (s + 1) * dim_y]
        for w_i in xrange(xs.shape[0]):
            w = i2w[np.argmax(xs[w_i, :])]
            if w == "<eoss>":
                break
            elif w_i != 0 and w == "<soss>":
                break
            print w.decode('utf-8')," ",
        print "\n"




def processing(x_path, y_path, threshold, get_num_start, get_num_end, batch_size = 1):
    X_seqs = []
    y_seqs = []
    i2w = {}
    w2i = {}
    lines_x = []
    lines_y = []
    tf = {}


    f_x = open(curr_path + "/" + x_path, "r")
    f_y = open(curr_path + "/" + y_path, "r")

    for line_x, line_y in zip(f_x, f_y):
        line_x = line_x.strip('\n')
        line_y = line_y.strip('\n')
        words_x = line_x.split() + ["<eoss>"]
        words_y = line_y.split() + ["<eoss>"]

        lines_x.append(words_x)
        lines_y.append(words_y)

        for w in (words_x + words_y):
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
                tf[w] = 1
            else:
                tf[w] += 1
    f_x.flush()
    f_y.flush()

    f_x.close()
    f_y.close()

    del f_x
    del f_y

    final_i2w = {}
    final_w2i = {}

    for word, num in tf.iteritems():
        if num > threshold:
            final_i2w[len(final_w2i)] = word
            final_w2i[word] = len(final_w2i)


    final_i2w[len(final_w2i)] = "<UNknown>"
    final_w2i["<UNknown>"] = len(final_w2i)

    i2w = final_i2w
    w2i = final_w2i
    # print len(i2w)
    del final_i2w
    del final_w2i

    assert len(lines_x) == len(lines_y)

    # for i in xrange(0, len(lines_x)):
    for i in xrange(get_num_start, get_num_end):
        # print i
        x_words = lines_x[i]
        y_words = lines_y[i]


        x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
        y = np.zeros((len(y_words), len(w2i)), dtype = theano.config.floatX)

        

        for j in range(0, len(x_words)):
            if x_words[j] in w2i:
                x[j, w2i[x_words[j]]] = 1
            else:
                # x_n += 1
                x[j, w2i["<UNknown>"]] = 1
        for k in range(0, len(y_words)):
            if y_words[k] in w2i:
                y[k, w2i[y_words[k]]] = 1
            else:
                y[k, w2i["<UNknown>"]] = 1
        
        X_seqs.append(x)
        y_seqs.append(y)

        del x
        del y

    data_x_y = batch_sequences(X_seqs, y_seqs, i2w, w2i, batch_size)

    return X_seqs, y_seqs, i2w, w2i, tf, data_x_y

def new_processing(x_path, y_path, w2i, i2w, get_num_start, get_num_end, batch_size = 1):
    X_seqs = []
    y_seqs = []

    lines_x = []
    lines_y = []

    f_x = open(curr_path + "/" + x_path, "r")
    f_y = open(curr_path + "/" + y_path, "r")

    for line_x, line_y in zip(f_x, f_y):
        line_x = line_x.strip('\n')
        line_y = line_y.strip('\n')
        words_x = line_x.split() + ["<eoss>"]
        words_y = line_y.split() + ["<eoss>"]

        lines_x.append(words_x)
        lines_y.append(words_y)
    f_x.flush()
    f_y.flush()

    f_x.close()
    f_y.close()

    del f_x
    del f_y


    assert len(lines_x) == len(lines_y)

    # for i in xrange(0, len(lines_x)):
    for i in xrange(get_num_start, get_num_end):
        # print i
        x_words = lines_x[i]
        y_words = lines_y[i]


        x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
        y = np.zeros((len(y_words), len(w2i)), dtype = theano.config.floatX)

        for j in xrange(0, len(x_words)):
            if x_words[j] in w2i:
                x[j, w2i[x_words[j]]] = 1
            else:
                # x_n += 1
                x[j, w2i["<UNknown>"]] = 1
        for k in xrange(0, len(y_words)):
            if y_words[k] in w2i:
                y[k, w2i[y_words[k]]] = 1
            else:
                y[k, w2i["<UNknown>"]] = 1
        
        X_seqs.append(x)
        y_seqs.append(y)

        del x
        del y

    data_x_y = batch_sequences(X_seqs, y_seqs, i2w, w2i, batch_size)

    return X_seqs, y_seqs, i2w, w2i, data_x_y


def test_processing(path, i2w, w2i, batch_size):
    f = open(path)
    X_seqs = []
    for line in f:
        # print i
        x_words = line.split() + ["<eoss>"]

        x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)

        for j in range(0, len(x_words)):
            if x_words[j] in w2i:
                x[j, w2i[x_words[j]]] = 1
            else:
                # x_n += 1
                x[j, w2i["<UNknown>"]] = 1

        X_seqs.append(x)
    f.close()
    data_x_y = batch_sequences(X_seqs, X_seqs, i2w, w2i, batch_size)
    return data_x_y

def test_sentence_input_processing(sentence, i2w, w2i, batch_size):
    x_words = sentence.split() + ["<eoss>"]
    x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
    for j in range(0, len(x_words)):
        if x_words[j] in w2i:
            x[j, w2i[x_words[j]]] = 1
        else:
            x[j, w2i["<UNknown>"]] = 1

    X_seqs = [x for i in range(batch_size)]
    data_x_y = batch_sequences(X_seqs, X_seqs, i2w, w2i, batch_size)
    return data_x_y

def test_sentence_input_processing_long(sentence, i2w, w2i, yl, batch_size):
    x_words = sentence.split() + ["<eoss>"]
    x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
    y = np.zeros((yl, len(w2i)), dtype = theano.config.floatX)
    for j in range(0, len(x_words)):
        if x_words[j] in w2i:
            x[j, w2i[x_words[j]]] = 1
        else:
            x[j, w2i["<UNknown>"]] = 1

    X_seqs = [x for i in range(batch_size)]
    Y_seqs = [y for i in range(batch_size)]
    data_x_y = batch_sequences(X_seqs, Y_seqs, i2w, w2i, batch_size)
    return data_x_y

def test_processing_long(path, i2w, w2i, yl, batch_size):
    f = open(path)
    X_seqs = []
    Y_seqs = []
    for line in f:
        # print i
        x_words = line.split() + ["<eoss>"]

        x = np.zeros((len(x_words), len(w2i)), dtype = theano.config.floatX)
        y = np.zeros((yl, len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(x_words)):
            if x_words[j] in w2i:
                x[j, w2i[x_words[j]]] = 1
            else:
                # x_n += 1
                x[j, w2i["<UNknown>"]] = 1

        X_seqs.append(x)
        Y_seqs.append(y)
    f.close()

    data_x_y = batch_sequences(X_seqs, Y_seqs, i2w, w2i, batch_size)
    return data_x_y





def batch_sequences(x_seqs, y_seqs, i2w, w2i, batch_size):
    assert len(x_seqs) == len(y_seqs)

    assert len(i2w) == len(w2i)

    data_x_y = {}
    batch_x = []
    batch_y = []
    x_seqs_len = []
    y_seqs_len = []

    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)

    for i in xrange(len(x_seqs)):
        xs = x_seqs[i]
        ys = y_seqs[i]

        X = xs[0 : len(xs), ]
        Y = ys[0 : len(ys), ]


        batch_x.append(X)
        batch_y.append(Y)

        x_seqs_len.append(X.shape[0])
        y_seqs_len.append(Y.shape[0])


        if len(batch_x) == batch_size or (i == len(x_seqs) - 1):
            x_max_len = np.max(x_seqs_len)
            y_max_len = np.max(y_seqs_len)

            mask_x = np.zeros((x_max_len, len(batch_x)), dtype = theano.config.floatX)
            mask_y = np.zeros((y_max_len, len(batch_y)), dtype = theano.config.floatX)

            concat_X = np.zeros((x_max_len, len(batch_x) * dim), dtype = theano.config.floatX)
            concat_Y = np.zeros((y_max_len, len(batch_y) * dim), dtype = theano.config.floatX)

            assert len(batch_x) == len(batch_y)
            
            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                Y = batch_y[b_i]

                mask_x[0 : X.shape[0], b_i] = 1
                mask_y[0 : Y.shape[0], b_i] = 1

                for r in xrange(x_max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                for rr in xrange(y_max_len - Y.shape[0]):
                    Y = np.concatenate((Y, zeros_m), axis=0)

                concat_X[:, b_i * dim : (b_i + 1) * dim] = X
                concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y

            data_x_y[batch_id] = [concat_X, mask_x, concat_Y, mask_y, len(batch_x)]
            batch_x = []
            batch_y = []

            x_seqs_len = []
            y_seqs_len = []
            batch_id += 1

    return data_x_y



