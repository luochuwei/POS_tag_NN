#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 17/03/2016
#    Usage: bleu
#
####################################################


from nltk.translate.bleu_score import *
import cPickle
import sys, os
import numpy as np
import re
import os

def print_bleu_normal(candidate_dic, reference_dic):
    bleu_list = []
    for i in candidate_dic:
        if len(candidate_dic[i]) < 4:
            bleu_list.append(0.0)
        else:
            bleu_list.append(sentence_bleu(reference_dic[i], candidate_dic[i]))

    # print "Bleu is", sum(bleu_list)/len(bleu_list)
    return float(sum(bleu_list))/len(bleu_list), bleu_list

def print_bleu_normal_batch(candidate_dic, reference_dic, start_num):
    bleu_list = []
    for i in candidate_dic:
        if len(candidate_dic[i]) < 4:
            w = [1.0/len(candidate_dic[i]) for i in range(len(candidate_dic[i]))]
            bleu_list.append(sentence_bleu(reference_dic[start_num+i], candidate_dic[i], weights=w))
        else:
            bleu_list.append(sentence_bleu(reference_dic[start_num+i], candidate_dic[i]))

    # print "Bleu is", sum(bleu_list)/len(bleu_list)
    return float(sum(bleu_list))/len(bleu_list), bleu_list




def get_candidate(path):
    nn = open(path)
    candidate_dic = {}
    n=0
    for line in nn:
        if len(line.strip()) > 1:
            candidate_dic[n] = line.strip().replace('<UNknown>', '').split()
            n += 1

    nn.close()
    return candidate_dic



# reference_dic = cPickle.load(open(r'reference_dic.pkl', 'rb'))



def print_bleu(candidate_dic, reference_dic, ngram):
    w = [1/float(ngram) for i in range(ngram)]
    sf = SmoothingFunction()
    # weights = [0.3, 0.7]
    bleu_list = []
    for i in candidate_dic:
        # print i
        if len(candidate_dic[i]) < len(w):
            bleu_list.append(0.0)
        else:
            # print i
            bleu_list.append(sentence_bleu(reference_dic[i], candidate_dic[i], weights=w))

    print "Bleu", len(w)," is", sum(bleu_list)/len(bleu_list)
    return sum(bleu_list)/len(bleu_list), bleu_list



# candidate_dic = get_candidate(r"output0422_small.txt")

# print_bleu(candidate_dic, reference_dic, 4)
# print_bleu(candidate_dic, reference_dic, 3)
# print_bleu(candidate_dic, reference_dic, 2)
# print_bleu(candidate_dic, reference_dic, 1)


# print_bleu_normal(candidate_dic, reference_dic)
# print len(candidate_dic)