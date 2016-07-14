#!/bin/python

# in consistence with the extract output embeddings json file

import sys
import json
import numpy as np
from sklearn.manifold import TSNE
configs = {"e1": "output_embeddings.json", "e2": "output_embeddings_2d.json",
           "step": "4",  "q": "", "range": "1.0", "which": "0", "number": "100", "repeat": "0", "print": "5"}
# q is "sent_index-arg_index-word_index"; range means half of the length of square/ number means the first nearest ones

def main():
    # read configurations
    for i in sys.argv:
        them = i.split(":")
        if len(them)==2:
            configs[them[0]] = them[1]
    print(configs)
    # dispatch
    # paramters
    v_query = configs["q"]
    v_range = float(configs["range"])
    v_num = int(configs["number"])
    v_which = int(configs["which"])     # which one, 0/1/2->cnn/enhanced/w
    v_repeat = bool(int(configs["repeat"]))
    v_step = int(configs["step"])
    v_print_till = int(configs["print"])
    if v_step==1:
        # tranform to 2d with t-sne
        with open(configs["e1"]) as f:
            all_them = [json.loads(s) for s in f]   # [sent_index, arg_index, word_index, word, sentence_words, [cnn], [enhanced], [w]]
        # t-sne
        print("- t-sne")
        array_w1 = np.array([z[5] for z in all_them])   # cnn
        array_w2 = np.array([z[6] for z in all_them])   # enhanced
        array_w3 = np.array([z[7] for z in all_them])   # w
        array_w4 = np.array([z[6][:50+50] for z in all_them])   #lstm
        model_w1 = TSNE(n_components=2, random_state=0)
        model_w2 = TSNE(n_components=2, random_state=0)
        model_w3 = TSNE(n_components=2, random_state=0)
        model_w4 = TSNE(n_components=2, random_state=0)
        array2d_w1 = model_w1.fit_transform(array_w1).tolist()
        array2d_w2 = model_w2.fit_transform(array_w2).tolist()
        array2d_w3 = model_w3.fit_transform(array_w3).tolist()
        array2d_w4 = model_w4.fit_transform(array_w4).tolist()
        # store
        all_them_v2 = []
        for i, one in enumerate(all_them):
            ind = "%s-%s-%s" % (one[0], one[1], one[2])
            all_them_v2.append([ind, one[3], one[4], array2d_w1[i], array2d_w2[i], array2d_w3[i], array2d_w4[i]])
        print("- write")
        with open(configs["e2"], "w") as f:
            for one in all_them_v2:
                f.write(json.dumps(one)+"\n")
    elif v_step==2 or v_step==3:
        with open(configs["e2"]) as f:
            all_them = [json.loads(s) for s in f]
        all_them_dict = {}
        for one in all_them:    # the-index-dict
            all_them_dict[one[0]] = one
        # find them all
        if v_query in all_them_dict:
            one = all_them_dict[v_query]
            place0 = one[3+v_which]
            print("Find %s: %s, finding %d at %s." % (v_query, one, v_which, place0))
            all_find = []
            all_words = set()
            if v_step == 2:
                for it in all_them:
                    if not v_repeat and (it[1] == one[1] or (it[1] in all_words)):
                        continue
                    place1 = it[3+v_which]
                    if abs(place0[0]-place1[0]) < v_range and abs(place0[1]-place1[1]) < v_range:
                        all_find.append(it)
                        all_words.add(it[1])
            elif v_step == 3:
                for it in all_them:
                    if not v_repeat and (it[1] == one[1] or (it[1] in all_words)):
                        continue
                    all_find.append(it)
                    all_words.add(it[1])    # add the first one
                all_find.sort(key=lambda x: ((x[3+v_which][0]-place0[0])**2+(x[3+v_which][1]-place0[1])**2))
                all_find = all_find[:v_num]
            print("Find all: %d:" % len(all_find))
            for one_it in all_find:
                print(one_it[:v_print_till])
        else:
            print("%s not found." % v_query)
    elif v_step==4:
        # get them all
        with open(configs["e1"]) as f:
            all_them = [json.loads(s) for s in f]
            for tmp_one in all_them:
                tmp_one.append(tmp_one[6][:50+50])
        all_them_dict = {}
        for one in all_them:    # the-index-dict
            ind = "%s-%s-%s" % (one[0], one[1], one[2])
            all_them_dict[ind] = one
        # find them all
        if v_query in all_them_dict:
            one = all_them_dict[v_query]
            place0 = one[2+3+v_which]
            print("Find %s: %s, finding %d." % (v_query, one[:5], v_which))
            all_find = []
            all_words = set()
            for it in all_them:
                if not v_repeat and (it[2+1] == one[2+1] or (it[2+1] in all_words)):
                    continue
                all_find.append(it)
                all_words.add(it[2+1])    # add the first one
                all_find.sort(key=lambda x: (np.sum((np.array(x[2+3+v_which]) - np.array(place0)) ** 2)))
                all_find = all_find[:v_num]
            print("Find all: %d:" % len(all_find))
            for one_it in all_find:
                print(one_it[:v_print_till])
        else:
            print("%s not found." % v_query)
    print("--------------------------------------------------")

main()

