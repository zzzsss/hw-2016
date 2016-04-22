#!/bin/python

import json
import sys

# 1. load data
def get_noexp_conll(ff):
    with open(ff) as f:
        train_data = [json.loads(i) for i in f]
    # implicit_data = [i for i in train_data if i["Type"] == "Implicit"]  # nope
    noexp_data = [i for i in train_data if len(i["Connective"]["TokenList"]) == 0]
    return noexp_data

# 2. vocab and relations --- only for train
def get_rvocab(data):
    r = {}
    for d in data:
        sl = d["Sense"]
        for s in sl:
            if s not in r:
                r[s] = len(r)
    print("- rvocab: ", len(r), "//", r)
    return r

def get_wvocab(data, thres=2):
    SPECIAL_TOKENS = ["<BLANK>", "<unk>", "<s>", "</s>"]  # impossible tokens in corpus
    # data is list of conll-relation, return a dictionary
    d = {}  # count dictionary
    for one in data:
        for w in one["Arg1"]["RawText"].split()+one["Arg2"]["RawText"].split():
            if w not in d:
                d[w] = 1
            else:
                d[w] += 1
    # then
    m = {}
    for t in SPECIAL_TOKENS:  # the <BLANK> is always 0 maybe meaning padding
        m[t] = len(m)
    for t in d:
        if(d[t] >= thres):
            m[t] = len(m)
    print("- wvocab: ", len(m))
    return m

# 3. get data as inputs, both for training and testing
def get_data(noexp_data, wvocab, rvocab):
    # list of [[arg1-index...], [arg2-index...], rel-index]
    def TMP_get(x):
        if x not in wvocab:
            return wvocab['<unk>']
        return wvocab[x]
    def TMP_get_lendict(l):
        d = {}
        for i in l:
            if len(i) not in d:
                d[len(i)] = 1
            else:
                d[len(i)] += 1
        return d
    ret = []
    for d in noexp_data:
        arg1 = [TMP_get(w) for w in d["Arg1"]["RawText"].split()]
        arg2 = [TMP_get(w) for w in d["Arg2"]["RawText"].split()]
        if len(arg2) > 200:
            pass
        try:
            rel = rvocab[d["Sense"][0]]     # only use sense 0
        except:
            rel = 'EntRel'   # just a placeholder
        ret.append([arg1, arg2, rel])
    print("- len for arg1: ", TMP_get_lendict([x[0] for x in ret]))
    print("- len for arg2: ", TMP_get_lendict([x[1] for x in ret]))
    print("-Number:", len(ret))
    return ret

def output_final(noexp_data, results, fname):
    # noexp_data is from get_noexp_conll, results is the list of name
    assert len(noexp_data) == len(results)
    with open(fname, "w") as f:
        for d, r in zip(noexp_data, results):
            z = {}
            z['Connective'] = {'TokenList': []}
            z['DocID'] = d['DocID']
            z['Sense'] = [r]
            z['Type'] = 'Implicit'
            if(r == "EntRel"):
                z['Type'] = 'EntRel'
            z['Arg1'] = {'TokenList': [x[2] for x in d['Arg1']['TokenList']]}
            z['Arg2'] = {'TokenList': [x[2] for x in d['Arg2']['TokenList']]}
            f.write(json.dumps(z) + "\n")
        print("write over to ", fname)

import sys
sys.path.append(".")
from conll_model import bow_model
from conll_model import cnn_model

the_models = [bow_model, cnn_model]

#### train ####
def main():
    # get data
    train_file = "conll16st-en-01-12-16-train/relations.json"
    dev_file = "conll16st-en-01-12-16-dev/relations.json"
    train_noexp_data = get_noexp_conll(train_file)
    dev_noexp_data = get_noexp_conll(dev_file)
    rvocab = get_rvocab(train_noexp_data)
    wvocab = get_wvocab(train_noexp_data)
    train_inputs = get_data(train_noexp_data, wvocab, rvocab)
    dev_inputs = get_data(dev_noexp_data, wvocab, rvocab)
    # training and testing
    outside_info = {}
    model_id = 1    # default the second one: cnn
    for a in sys.argv[1:]:
        two = a.split(":")
        if(len(two) == 2):
            outside_info[two[0]] = two[1]
        elif(len(two) == 1):
            model_id = int(two[0])
    outside_info["max_features"] = len(wvocab)
    outside_info["odim"] = len(rvocab)
    outside_info["wvocab"] = wvocab
    outside_info["rvocab"] = rvocab
    m = the_models[model_id](outside_info)
    m.train_it(train_inputs, dev_inputs)
    print("--test dev:")
    dev_result = m.test_it(dev_inputs)   # test_it re-write the fields of the list
    # final
    rev_list = [None for i in range(len(rvocab))]
    for k in rvocab:
        rev_list[rvocab[k]] = k
    output_final(dev_noexp_data, [rev_list[i] for i in dev_result], "output-dev.json")

# to score: python conll_scorer/scorer.py conll16st-en-01-12-16-dev/relations.json output-dev.json

if __name__ == '__main__':
    main()

