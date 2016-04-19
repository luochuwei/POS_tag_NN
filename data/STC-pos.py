#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 18/04/2016
#    Usage: process stc data and get the pos tag
#
#######################################################

import os
from nltk.tag import StanfordPOSTagger
import cPickle


"""
StanfordPOSTagger example :

st = StanfordPOSTagger(model_filename = r'D:\stanford\stanford-postagger-full-2015-12-09\models\chinese-distsim.tagger', path_to_jar = r'D:\stanford\stanford-postagger-full-2015-12-09\stanford-postagger.jar')

s = st.tag('哈哈 EMC 这 也 能 加密'.decode('utf-8').split())

for i in s:
    print i[-1]

哈哈#IJ
这#PN
也#AD
能#VV
加密#VV
"""

def pos_tag(st, sentence):
    s = st.tag(sentence.decode('utf-8').split())
    word_tag_list = []
    tag_return = ""
    for item in s:
        w, tag = item[-1].split('#')
        word_tag_list.append((w, tag))
        tag_return += tag
        tag_return += ' '
    return word_tag_list, tag_return[:-1]


#初始化stanford Chinese pos tagger
st = StanfordPOSTagger(model_filename = r'D:\stanford\stanford-postagger-full-2015-12-09\models\chinese-distsim.tagger', path_to_jar = r'D:\stanford\stanford-postagger-full-2015-12-09\stanford-postagger.jar')

#file path
code_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
post_file = r'STC_data\new_train.post'
response_file = r'STC_data\new_train.response'

post = open(code_path + "/" + post_file)
response = open(code_path+ "/" +response_file)

post_tag_file = open(code_path+ "/"  + r'STC_data\post.tag', 'w')
response_tag_file = open(code_path+ "/"  + r'STC_data\response.tag', 'w')


pos_tag_dic = {}
#example: pos_tag_dic['NN'] = tag_dic
# tag_dic['老板'] = 11

for p, r in zip(post, response):
    p = p.strip()
    r = r.strip()
    p_wt, p_t = pos_tag(st, p)
    r_wt, r_t = pos_tag(st, r)
    post_tag_file.write(p_t + '\n')
    response_tag_file.write(r_t + '\n')

    for w,t in p_wt + r_wt:
        if t not in pos_tag_dic:
            pos_tag_dic[t] = {}
            pos_tag_dic[t][w] = 1
        else:
            if w not in pos_tag_dic[t]:
                pos_tag_dic[t][w] = 1
            else:
                pos_tag_dic[t][w] += 1


cPickle.dump(pos_tag_dic, open(code_path + "/" + r'STC_data\pos_tag_dic.pkl', 'wb'))

print "statistics of the STC dataset"
print "length of pos tagset : ", str(len(pos_tag_dic))

#most 5 frequent words in every tags
for t in pos_tag_dic:
    sorted_taglist = sorted(pos_tag_dic[t].iteritems(), key = lambda d:d[1], reverse = True)
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print t
    print "most 5 frequent words"
    for w, n in sorted_taglist[0:5]:
        print w, ":" ,str(n), "times"


print "done"


