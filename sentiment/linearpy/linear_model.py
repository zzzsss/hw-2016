import sys
import json
sys.path.append(".")
from svmutil import *


DEFAULT_OPTS = {"pre-not": 0, "pre-link": 0,
                "f-thres": 1, "f-unigram": 1, "f-bigram": 1, "f-trigram": 1}
OPTS_TRANS = {"pre-not": int, "pre-link": int,
                "f-thres": int, "f-unigram": int, "f-bigram": int, "f-trigram": int}
DICT_PUN = ",.;?!~"
DICT_NOT = "not no n't"
DICT_LINK = "although but however if otherwise unless so"


def add_count(d, s):
    if s not in d:
        d[s] = 1
    else:
        d[s] += 1


class model_svm:
    MNAME = "model.m"
    VNAME = "vocab.json"

    def __init__(self, info):
        self.model = None
        self.vocab = None
        self.opts = DEFAULT_OPTS
        opts = self.opts
        for k in info:
            if k in OPTS_TRANS:
                opts[k] = OPTS_TRANS[k](info[k])
            else:
                opts[k] = info[k]
        print("-- Final options are: %s" % str(opts))

    def train_it(self, train_inputs, dev_inputs):
        # 1. load options
        opts = self.opts
        # 2. build vocabs and get inputs
        trainx = self.get_features(train_inputs)
        trainy = [z[1] for z in train_inputs]       # the second one is class
        # 3. train svm
        prob = svm_problem(trainy, trainx)
        print("- Train svm with ", opts["EXTRA"])
        self.model = svm_train(prob, opts["EXTRA"])

    def save_it(self, dir_name):
        svm_save_model(dir_name+model_svm.MNAME, self.model)
        with open(dir_name+model_svm.VNAME, "w") as f:
            f.write(json.dumps([self.vocab, self.opts]))

    def load_it(self, dir_name):
        self.model = svm_load_model(dir_name+model_svm.MNAME)
        with open(dir_name+model_svm.VNAME, "r") as f:
            self.vocab, self.opts = json.loads(f.read())

    def test_it(self, inputs):
        devx = self.get_features(inputs, False)
        devy = [z[1] for z in inputs]       # the second one is class
        p_label, p_acc, _ = svm_predict(devy, devx, self.model)
        print("Test is %s" % p_acc[0])
        return p_label

    # the core function ...
    def get_features(self, data, train=True):
        # return trainx
        ret = []
        # step 1: optional pre-process
        for d in data:
            s = d[0]
            l = len(s)
            state_not = ""
            state_link = ""
            for i in range(l):
                if s[i] in DICT_NOT:
                    state_not += "not:"      # maybe notnot
                elif s[i] in DICT_LINK:
                    state_link = s[i]+":"
                elif s[i] in DICT_PUN:
                    state_not = ""
                    state_link = ""
                else:
                    if self.opts["pre-not"]:
                        s[i] += state_not
                    if self.opts["pre-link"]:
                        s[i] += state_link
        # step 2: get vocab (if train) and features
        # 2.1: get features and count
        if train:
            self.vocab = {}
        features = []      # list of set of strings
        for d in data:
            s = d[0]
            l = len(s)
            fs = set()
            # get features
            for i in range(l):
                if self.opts["f-unigram"]:
                    fs.add(s[i])
                    if train:
                        add_count(self.vocab, s[i])
                if self.opts["f-bigram"] and i < l-1:
                    tmps = s[i]+"|"+s[i+1]
                    fs.add(tmps)
                    if train:
                        add_count(self.vocab, tmps)
                if self.opts["f-trigram"] and i < l-2:
                    tmps = s[i]+"|"+s[i+1]+"|"+s[i+2]
                    fs.add(tmps)
                    if train:
                        add_count(self.vocab, tmps)
            features.append(fs)
        # 2.2 remove low-freq features
        if train and self.opts["f-thres"]:
            tmp_vocab = self.vocab
            self.vocab = {}
            for v in tmp_vocab:
                if tmp_vocab[v] > self.opts["f-thres"]:
                    self.vocab[v] = len(self.vocab)
            print("Build features from %d to %d." % (len(tmp_vocab), len(self.vocab)))
        # 2.3 return the indexes
        for fs in features:
            # only 0-1 features
            one_fi = [self.vocab[s] for s in fs if s in self.vocab]
            one_fi.sort()
            one_fd = {}
            for i in one_fi:
                one_fd[i] = 1
            ret.append(one_fd)
        return ret
