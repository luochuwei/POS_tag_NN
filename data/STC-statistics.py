#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 18/04/2016
#    Usage: statistic of stc pos data
#
#######################################################


import os
import cPickle

code_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

post_tag_file = r'STC_data\post_tag'
response_tag_file = r'STC_data\response_tag'

post_tag = open(code_path + "/" + post_tag_file)
response_tag = open(code_path + "/" + response_tag_file)

pos_tag_dic = {}

for pt, rt in zip(post_tag, response_tag):
    pt = pt.strip()
    rt = rt.strip()
    p_wt = [(item.split("#")[0],item.split("#")[1]) for item in pt.split()]
    r_wt = [(item.split("#")[0],item.split("#")[1]) for item in rt.split()]

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
