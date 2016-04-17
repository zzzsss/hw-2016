#!/bin/python

# read the data from the sentiment treebank

f_dir = "./stanfordSentimentTreebank/"
f_datasplit = f_dir + "datasetSplit.txt"
f_labels = f_dir + "sentiment_labels.txt"
f_dict = f_dir + "dictionary.txt"
f_sent = f_dir + "datasetSentences.txt"

# Part1: the data reading part

class Sent:
    def __init__(self, words, score):
        self.words = words
        self.score = score
        self.indexes = None

    def get_words(self):
        return self.words

    def get_score(self):
        return self.score

    def get_indexes(self):
        if self.indexes is None:
            raise 1
        return self.indexes

    def __str__(self):
        s = str(self.score)
        for w in self.words:
            s = s + " " + w
        return s

    # after build the dictionary
    def map_index(self, d):
        self.indexes = [d[w] for w in self.words]

    # get data-format for model in batch mode - first dim is batch
    #  -> return dx(list of list of int), dy(list of float)
    @staticmethod
    def get_modeldata(sents):
        dx = []
        dy = []
        for s in sents:
            dx.append([i for i in s.get_indexes()])
            dy.append(s.get_score())
        return dx, dy

def TMP_delete_nonascii(x):
    if(isinstance(x, str)):
        ret = ""
        for c in x:
            i = ord(c)
            if(i >= 0 and i<=255):
                ret += c
        return ret
    elif(isinstance(x, list)):
        return [TMP_delete_nonascii(i) for i in x]
    else:
        raise 1

def TMP_trans(x):
    ret = []
    dd = {"-LRB-": "(", "-RRB-": ")"}
    for s in x:
        if(s in dd):
            ret.append(dd[s])
        else:
            ret.append(s)
    return ret

def read_sentences():
    # return (train, test, dev), and each is list of Sent
    with open(f_sent, "r") as f:
        list_raw = [TMP_delete_nonascii(l.split()[1:]) for l in f.readlines()[1:]]   # ignore the first line
    phrase_dict = {}
    with open(f_dict, "r") as f:
        for l in f.readlines():
            s = l.split("|")
            s0 = TMP_delete_nonascii("".join(s[0].split()))
            s1 = int(s[1])
            phrase_dict[s0] = s1
    with open(f_labels, "r") as f:
        list_pscore = [float(l.split("|")[1]) for l in f.readlines()[1:]]
    # -> get root score
    list_all = []
    miss_all = []
    for sent in list_raw:
        roots = "".join(sent)
        try:
            index = phrase_dict[roots]
        except:
            roots = "".join(TMP_trans(sent)) # convert the '()'
            try:
                index = phrase_dict[roots]
            except:
                miss_all.append(sent)
        score = list_pscore[index]
        list_all.append(Sent(sent, score))
    # split them
    ret = ([], [], [])
    with open(f_datasplit, "r") as f:
        list_split = [int(l.split(",")[1])-1 for l in f.readlines()[1:]]
        for i, s in enumerate(list_all):
            index = list_split[i]
            ret[index].append(s)
    print(len(ret[0]), len(ret[1]), len(ret[2]))
    return ret


# Part2: the dictionary

class Dict:
    SPECIAL_TOKENS = ["<BLANK>", "<unk>", "<s>", "</s>"]  # impossible tokens in corpus

    def __init__(self, m):
        self.map = m

    def __getitem__(self, item):
        if item in self.map:
            return self.map[item]
        else:
            return self.map["<unk>"]

    def __len__(self):
        return len(self.map)

    def keys(self):
        return self.map.keys()

    @staticmethod
    def construct_d(sents, thres=2):
        # remain only freq >= thres
        counts = {}
        for s in sents:
            for w in s.get_words():
                if w in counts:
                    counts[w] += 1
                else:
                    counts[w] = 1
        m = {}
        for t in Dict.SPECIAL_TOKENS:  # the <BLANK> is always 0 maybe meaning padding
            m[t] = len(m)
        for t in counts:
            if(counts[t] >= thres):
                m[t] = len(m)
        return Dict(m)

# write to train, dev, test
def write_format_data(prefix=""):
    train, test, dev = read_sentences()
    # score, sentences ...
    for d, name in zip([train, test, dev], ['train', 'test', 'dev']):
        with open(prefix+name+".txt", 'w') as f:
            for i in d:
                f.write(str(i))
                f.write("\n")



