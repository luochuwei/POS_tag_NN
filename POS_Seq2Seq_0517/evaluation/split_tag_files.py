#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 16/05/2016
#    Usage: split tag file into two files
#
####################################################



def split_files(in_file_path, text_path, tag_path):
    f = open(in_file_path)
    ftext = open(text_path, 'w')
    ftag = open(tag_path, 'w')
    for line in f:
        line_list = line.strip().split()
        for i in line_list:
            word, tag = i.split("#")
            ftext.write(word+' ')
            ftag.write(tag+' ')
        ftext.write('\n')
        ftag.write('\n')
    f.close()
    ftext.close()
    ftag.close()
    print "done"



# split_files(r'validation_file.tag', r'validation_file-0517.txt', r'validation_file_only_tag-0517.txt')

split_files(r'testfile-with-tag.txt', 'testfile-post.txt', 'testfile-only-tag.txt')

# split_files('basketball-response_tag.txt', 'basketball-response_0527.txt', 'basketball-response-tag_0527.txt')
