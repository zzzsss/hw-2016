import sys
sys.path.append(".")
from svmutil import *
import numpy as np
from conn_head_mapper import DEFAULT_MAPPING
import json

# some common procedures
def get_rvocab(data):
    r = {}
    for d in data:
        sl = d["Sense"]
        for s in sl:
            if s not in r:
                r[s] = len(r)
    print("- rvocab: ", len(r), "//", r)
    # reverse
    tmp = list(r.keys())
    for s in tmp:
        r[str(r[s])] = s
    return r

# from the "get_data" function
def get_features(data,  opts, wvocab=None):
    # return (the-dict, [[feature-set]])
    # consider only connective
    to_build = False
    if wvocab is None:
        to_build = True
        wvocab = {}
    # what kind of feature to consider --- please don't <list, list>
    fset = []
    fthem = ['w','p','r','hw','hp',  'sw','sp','sr',  'cw','cp','cr']
    for i in opts:
        tmpl = i.split("+")
        if not all([(z in fthem) for z in tmpl]):
            continue
        if(int(opts[i])):
            fset.append(tmpl)
    print("Feature template: "+str(fset))
    # collect feature strings
    f_ret = []
    for one in data:
        # get them ..
        one_collect = set()
        for i in range(len(one['Connective']['w'])):    # each conn word
            for fl in fset:     # each feature template
                tmp_prefix = "+".join(fl)
                tmp_them = [one['Connective'][zz][i] for zz in fl]
                tmp_build = [""]
                for zz in tmp_them:
                    if isinstance(zz, str):
                        tmp_build = [xx+"|"+zz for xx in tmp_build]
                    elif isinstance(zz, list):
                        if len(tmp_build) > 1:
                            print("Warning, not-appropriate template: ", tmp_prefix)
                        tmp_build = [tmp_build[0]+"|"+xx for xx in zz]
                for zz in tmp_build:
                    one_collect.add(tmp_prefix+zz)
        # for feat in one['Connective']:
        #     if int(opts[feat]):     # if opened by option
        #         for temp_one in one['Connective'][feat]:
        #             if isinstance(temp_one, str):
        #                 one_collect.add(feat+"|"+temp_one)
        #             elif isinstance(temp_one, list):
        #                 for ttt in temp_one:
        #                     one_collect.add(feat+"|"+ttt)
        # build ??
        if(to_build):
            for s in one_collect:
                if s not in wvocab:
                    wvocab[s] = len(wvocab)
        # get them
        one_fl = [wvocab[s] for s in one_collect if s in wvocab]
        one_fl.sort()
        one_fd = {}
        for i in one_fl:
            one_fd[i] = 1
        f_ret.append(one_fd)
    if(to_build):
        print("Build features %d" % len(wvocab))
    return wvocab, f_ret

def get_yindex(rvocab, data):
    return [rvocab[i['Sense'][0]] for i in data]

DEFAULT_OPS = {'w':1,'p':1,'r':1,'hw':0,'hp':1,'sw':0,'sp':0,'sr':0,'cw':0,'cp':0,'cr':0,
                "EXTRA":"",
               'w+hp':1, 'p+hp':1, 'r+hp':1     # useful ones maybe
               }

class model_svm:
    MNAME = "/model.m"
    VNAME = "/vocab.json"
    def __init__(self, info={}):
        self.model = None
        self.rvocab = None
        self.wvocab = None
        self.opts = DEFAULT_OPS
        opts = self.opts
        for k in info:
            opts[k] = info[k]
        print("-- Final options are: %s" % opts)

    def train_it(self, train_inputs, dev_inputs):
        # 1. load options
        opts = self.opts
        # 2. build vocabs and get inputs
        self.rvocab = get_rvocab(train_inputs)
        self.wvocab, trainx = get_features(train_inputs, self.opts)
        trainy = get_yindex(self.rvocab, train_inputs)
        # 3. train svm
        prob = svm_problem(trainy, trainx)
        print("- Train svm with ", opts["EXTRA"])
        self.model = svm_train(prob, opts["EXTRA"])

    def save_it(self, dir_name):
        svm_save_model(dir_name+model_svm.MNAME, self.model)
        with open(dir_name+model_svm.VNAME, "w") as f:
            f.write(json.dumps([self.rvocab, self.wvocab, self.opts]))

    def load_it(self, dir_name):
        self.model = svm_load_model(dir_name+model_svm.MNAME)
        with open(dir_name+model_svm.VNAME, "r") as f:
            self.rvocab, self.wvocab, self.opts = json.loads(f.read())

    def test_it(self, inputs):
        _, devx = get_features(inputs, self.opts, self.wvocab)
        devy = get_yindex(self.rvocab, inputs)
        p_label, p_acc, _ = svm_predict(devy, devx, self.model)
        print("Test is %s" % p_acc[0])
        return [self.rvocab[str(int(i))] for i in p_label]

class model_baseline:
    def __init__(self, info={}):
        self.model = {}
        self.wmap = {}

    def _convert(self, wl):
        # input conn word list
        s = "".join([x for x in wl]).lower()
        if s in self.wmap:
            return self.wmap[s]
        if wl[-1] in self.wmap:
            return self.wmap[wl[-1]]
        print("Warning: not found ", str(wl))
        return ""

    def train_it(self, train_inputs, dev_inputs):
        for m in DEFAULT_MAPPING:
            self.wmap["".join(m.split()).lower()] = DEFAULT_MAPPING[m]
        for m in self.wmap:
            self.model[self.wmap[m]] = {}
        for d in train_inputs:
            smap = self._convert(d['Connective']['w'])
            sense = d['Sense'][0]
            if sense not in self.model[smap]:
                self.model[smap][sense] = 1
            else:
                self.model[smap][sense] += 1
        # count
        for m in self.model:
            them = self.model[m]
            max_them = 0
            all_sense = list(them.keys())
            for s in all_sense:
                if them[s] > max_them:
                    max_them = them[s]
                    them[0] = s
        self.model[""] = {0: "Nope"}    # give up those
        # ok
        print(self.model)

    def test_it(self, inputs):
        ret = []
        correct = 0
        for d in inputs:
            smap = self._convert(d['Connective']['w'])
            this_s = self.model[smap][0]
            if this_s == d['Sense'][0]:
                correct += 1
            ret.append(this_s)
        print("Test acc temp is:", correct/len(ret))
        return ret

the_models = [model_svm, model_baseline]


