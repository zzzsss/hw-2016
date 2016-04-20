#!/bin/python

import json
import sys
import codecs
sys.path.append(".")
from conn_head_mapper import DEFAULT_MAPPING

# analysis and the driver for training and testing

# 1. load data --- from relation.json !!! only for training
def get_exp_conll(ff):
    with open(ff) as f:
        train_data = [json.loads(i) for i in f]
    # implicit_data = [i for i in train_data if i["Type"] == "Implicit"]  # nope
    # exp_data = [i for i in train_data if len(i["Connective"]["TokenList"]) != 0]
    exp_data = [i for i in train_data if i["Type"] == "Explicit"]
    return exp_data

def get_pdict(ff):
    parse_file = codecs.open(ff, encoding='utf8')
    en_parse_dict = json.load(parse_file)
    return en_parse_dict

# 2. deal with data --- only for train
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
        for w in one["Arg1"]["RawText"].split()+one["Arg2"]["RawText"].split()+one['Connective']['RawText']:
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

# 2'. analysis

def TMP_addone(d, one):
    one = str(one)
    if(one not in d):
        d[one] = 1
    else:
        d[one] += 1

def analysis_conn(data):
    # analysis the conn with mapping
    mapper = set()
    for k in DEFAULT_MAPPING:
        mapper.add(DEFAULT_MAPPING[k])
    mapper_str = {}     # mapper->{str:count}
    mapper_num = {}     # mapper->int
    mapper_hit = {}     # mapper->{sense:count}
    for m in mapper:
        mapper_str[m] = {}
        mapper_num[m] = 0
        mapper_hit[m] = {}
    # analysis
    unknown = []
    for d in data:
        s = d['Connective']['RawText']
        if s not in DEFAULT_MAPPING:
            unknown.append(s)
        else:
            m = DEFAULT_MAPPING[s]
            mapper_num[m] += 1
            TMP_addone(mapper_str[m], s)
            TMP_addone(mapper_hit[m], d['Sense'])
    # report
    print("All %d, unknown %d." % (len(data), len(unknown)))
    print("- Unknown is %s." % unknown)
    print("- Mapper num is %d:" % len(mapper))
    for m in mapper:
        print("-- %s: %s, %s, %s" % (m, mapper_num[m], mapper_str[m], mapper_hit[m]))

def get_data(data, parses):
    # deal with the dependency tree first --- find child and sibling
    # !! no dependency for punctuation ...
    for doc in parses:
        for one in parses[doc]['sentences']:
            tmp_len = len(one['words'])
            one["d-child"] = [[] for i in range(tmp_len + 1)]  # !! special for root
            one["d-parent"] = [i for i in range(tmp_len)]       # special for punctuation (self link, maybe no problem)
            one["d-relation"] = ['<zz-punc>' for i in range(tmp_len)]
            for item in one['dependencies']:
                head = int(item[1].split("-")[-1])      # head is index+1
                mod = int(item[2].split("-")[-1])-1     # mod is index
                one["d-child"][head].append(mod)    # index+1 map to index
                one["d-parent"][mod] = head         # index map to index+1
                one["d-relation"][mod] = item[0]    # index to str
    # get the features of connectives
    ret = []
    TMP_ROOT_TOKEN = "<ROOT>"
    for d in data:
        one = {}
        docId = d['DocID']
        one['Sense'] = d['Sense']
        for which in ['Arg1', 'Arg2', 'Connective']:
            # word, pos, dependency-relation, --- self, head, <sibling, child> (<> has multiple)
            one[which] = {'w': [], 'p': [], 'r': [], 'hw': [], 'hp': [],                # list of str
                          'sw': [], 'sp': [], 'sr': [], 'cw': [], 'cp': [], 'cr': []}   # list of list of str
            x = one[which]
            for i in d[which]['TokenList']:
                tmp_sent = i[3]
                tmp_index = i[4]
                x['w'].append(parses[docId]['sentences'][tmp_sent]['words'][tmp_index][0])
                x['p'].append(parses[docId]['sentences'][tmp_sent]['words'][tmp_index][1]['PartOfSpeech'])
                x['r'].append(parses[docId]['sentences'][tmp_sent]["d-relation"][tmp_index])
                hindex = parses[docId]['sentences'][tmp_sent]['d-parent'][tmp_index] - 1
                if(hindex < 0):
                    x['hw'].append(TMP_ROOT_TOKEN)
                    x['hp'].append(TMP_ROOT_TOKEN)
                else:
                    x['hw'].append(parses[docId]['sentences'][tmp_sent]['words'][hindex][0])
                    x['hp'].append(parses[docId]['sentences'][tmp_sent]['words'][hindex][1]['PartOfSpeech'])
                sindexes = parses[docId]['sentences'][tmp_sent]['d-child'][hindex+1]
                if tmp_index in sindexes:
                    sindexes.remove(tmp_index)  # remove self
                cindexes = parses[docId]['sentences'][tmp_sent]['d-child'][tmp_index+1]
                for li, name in zip([sindexes, cindexes], ['s', 'c']):
                    x[name+'w'].append([parses[docId]['sentences'][tmp_sent]['words'][zz][0] for zz in li])
                    x[name+'p'].append([parses[docId]['sentences'][tmp_sent]['words'][zz][1]['PartOfSpeech'] for zz in li])
                    x[name+'r'].append([parses[docId]['sentences'][tmp_sent]["d-relation"][zz] for zz in li])
        ret.append(one)
    return ret


def output_final(exp_data, results, fname):
    # noexp_data is from get_noexp_conll, results is the list of name
    assert len(exp_data) == len(results)
    with open(fname, "w") as f:
        for d, r in zip(exp_data, results):
            z = {}
            z['Connective'] = d['Connective']
            z['Connective']['TokenList'] = [x[2] for x in d['Connective']['TokenList']]
            z['DocID'] = d['DocID']
            z['Sense'] = [r]
            z['Type'] = 'Explicit'
            z['Arg1'] = {'TokenList': [x[2] for x in d['Arg1']['TokenList']]}
            z['Arg2'] = {'TokenList': [x[2] for x in d['Arg2']['TokenList']]}
            f.write(json.dumps(z) + "\n")
        print("write over to ", fname)

import sys
import os
sys.path.append(".")
from conll_exp_model import the_models

#### train ####
def main():
    # get data
    train_rel = get_exp_conll("train/relations.json")
    train_parse = get_pdict("train/parses.json")
    dev_rel = get_exp_conll("dev/relations.json")
    dev_parse = get_pdict("dev/parses.json")
    train_data = get_data(train_rel, train_parse)
    dev_data = get_data(dev_rel, dev_parse)
    outside_info = {}
    outside_info["EXTRA"] = "-q -t 0"
    model_id = 0    # first one
    for a in sys.argv[1:]:
        two = a.split(":")
        if(len(two) == 2):
            outside_info[two[0]] = two[1]
            if(two[0] == 'M'):
                model_id = int(two[1])
        else:
            outside_info["EXTRA"] += " "+a
    m = the_models[model_id](outside_info)
    m.train_it(train_data, dev_data)
    m.save_it("models")
    print("--test dev:")
    dev_result = m.test_it(dev_data)
    # final
    output_final(dev_rel, dev_result, "output-dev.json")
    # score
    os.system("python conll_scorer/scorer.py dev/relations.json output-dev.json")

## test on dev ##
def main2_svm():
    dev_rel = get_exp_conll("dev/relations.json")
    dev_parse = get_pdict("dev/parses.json")
    dev_data = get_data(dev_rel, dev_parse)
    m = the_models[0]()
    m.load_it("models")
    print("--test dev:")
    dev_result = m.test_it(dev_data)
    # final
    output_final(dev_rel, dev_result, "output-dev.json")
    # score
    os.system("python conll_scorer/scorer.py dev/relations.json output-dev.json")

# to score: python conll_scorer/scorer.py conll16st-en-01-12-16-dev/relations.json output-dev.json
if __name__ == '__main__':
    main()
    main2_svm()
