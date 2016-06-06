#!/bin/python3

import sys

# char_f = sys.argv[1]
char_f = "./anno_20120509_45000.txt"
tran_f = sys.argv[2]
out_f = sys.argv[3]

# 1. first load the char dependency file
char_dict = {}      # word -> [pos-list, head-list, label-list]
with open(char_f, 'r') as f:
    one_w = None
    one_l = None
    c_flag = False
    for i, l in enumerate(f):
        fields = l.strip().split()
        try:
            index = int(fields[0])
            c_flag = True
            if(index == 1):
                # start a new one
                one_w = fields[1]
                one_l = [[fields[2]], [int(fields[3])], [fields[4]]]
            else:
                one_w += fields[1]
                one_l[0].append(fields[2])
                one_l[1].append(fields[3])
                one_l[2].append(fields[4])
        except:
            if(c_flag):
                if one_w in char_dict:
                    print("Warning existing key: ", one_w)
                char_dict[one_w] = one_l
            c_flag = False
    if(c_flag):
        if one_w in char_dict:
            print("Warning existing key: ", one_w)
        char_dict[one_w] = one_l
print("Read char file and all ", len(char_dict))

# tranform
def TMP_transform(l, cdict):
    # return a new list
    pass

# 2. then transform the inputs
# CoNLL08 format [0:index, 1:word, ?:pos, ?:head, ?:label]
sent = [['root', 'root', -1, 'root']]
num = 0
with open(tran_f, 'r') as fin:
    with open(out_f, 'w') as fout:
        c_flag = False
        for i, l in enumerate(fin):
            fields = l.strip().split()
            try:
                index = int(fields[0])
                if(index == 1):
                    # start a new one
                    sent = [['root', 'root', -1, 'root']]
                sent.append([])
            except:
                if(c_flag):
                    # write out one
                    num += 1
                    pass
                c_flag = False
        # assume empty line in the end
        pass
print("Transform all with ", num)
