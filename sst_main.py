#!/bin/python

from sst import *

# part3: training and testing

import sys
sys.path.append(".")
from sst_model import bow_model
from sst_model import cnn_model

the_models = [bow_model, cnn_model]

def analyse_data(dx, dy):
    length_map = {}
    for x in dx:
        l = len(x)
        if l in length_map:
            length_map[l] += 1
        else:
            length_map[l] = 1
    print(length_map)


def main():
    # prepare data
    train, test, dev = read_sentences()
    wdic = Dict.construct_d(train)
    for s in train+test+dev:
        s.map_index(wdic)
    trainx, trainy = Sent.get_modeldata(train)
    testx, testy = Sent.get_modeldata(test)
    devx, devy = Sent.get_modeldata(dev)
    analyse_data(trainx, trainy)   ### max-len 52, set maxlen to 50 is fine
    # training and testing
    outside_info = {}
    model_id = 1    # default the second one: cnn
    for a in sys.argv[1:]:
        two = a.split(":")
        if(len(two) == 2):
            outside_info[two[0]] = two[1]
        elif(len(two) == 1):
            model_id = int(two[0])
    outside_info["max_features"] = len(wdic)
    outside_info["dictionary"] = wdic
    m = the_models[model_id](outside_info)
    m.train_it(trainx, trainy, devx, devy)
    print("--test dev:")
    m.test_it(devx, devy)
    print("--test test:")
    m.test_it(testx, testy)

if __name__ == '__main__':
    main()
