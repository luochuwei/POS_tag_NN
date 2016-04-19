#-*- coding:utf-8 -*-
#######################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 18/04/2016
#    Usage: draw stc pos data
#
#######################################################

import os
import cPickle
import matplotlib.pyplot as plt
import numpy as np


#statistics
code_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
pos_tag_dic = cPickle.load(open(code_path + "/" + r'STC_data\pos_tag_dic.pkl', 'rb'))

tag_num_of_words = {}
tag_class_of_words = {}

for tag in pos_tag_dic:
    tag_num_of_words[tag] = 0
    for word, num in pos_tag_dic[tag].iteritems():
        tag_num_of_words[tag] += num
    tag_class_of_words[tag] = len(pos_tag_dic[tag])


def autolabel(rects, ax):
    """attach text labels to the bar graph"""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height, '%d'%int(height), ha ='center', va = 'bottom')


#设置自动颜色
N = len(pos_tag_dic)

color_list = plt.cm.rainbow(np.linspace(0,1,N))

#画图
Fig = plt.figure(0)
ax1 = plt.subplot2grid((2,2),(0,0), colspan = 2)
ax2 = plt.subplot2grid((2,2),(1,0), colspan = 2)

#各个tag下面一共有多少个词
plt.sca(ax1)
plt.title("Num of words in each tag")

ind = np.arange(N)
p1 = plt.bar(ind, height = [j for i,j in tag_num_of_words.iteritems()], align = 'center', color = color_list)
plt.xticks(ind, [i for i in tag_num_of_words])
plt.ylabel("Num of words")
autolabel(p1, ax1)


#各个tag下面一共有多少种类的词
plt.sca(ax2)
plt.title("class of words in each tag")

ind = np.arange(N)
p2 = plt.bar(ind, height = [j for i,j in tag_class_of_words.iteritems()], align = 'center', color = color_list)
plt.xticks(ind, [i for i in tag_class_of_words])
plt.ylabel("class of words")
autolabel(p2, ax2)

plt.show()
Fig.savefig("visualizations.png", format = "png")