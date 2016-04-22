import numpy as np
from keras.models import Sequential, Graph
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import Adagrad
from keras import backend as K

np.random.seed(12345)


class the_model:
    DEFAULT_OPS = {"max_features": None, "embedding_dims": 100, "maxlen": 120, "nb_filter": 256, "filter_length": 3, "activation": 'tanh', "pool_length": 120, "hidden_dims": 128,  "batch_size": 128, "iters": 10, "lr": 0.01, "dropout": 0.3,
                   "embed_l": None, "embed_w": None, "embed_unk_rand": 0.1,
                   "namep": "conll_noexp"}

    def __init__(self, info={}):
        self.model = None
        self.opts = the_model.DEFAULT_OPS
        self.configs = dict()
        opts = self.opts
        for k in info:
            opts[k] = info[k]
        print("-- Final options are:")
        for k in self.opts:
            if k == "wvocab":
                print("wvocab: ...")
            else:
                print(k, self.opts[k])

    def preprocess(self, dx1, dx2, dy):
        print("Preprocess data.")
        px1 = sequence.pad_sequences(dx1, maxlen=self.opts["maxlen"])
        px1 = np.asarray(px1)
        px2 = sequence.pad_sequences(dx2, maxlen=self.opts["maxlen"])
        px2 = np.asarray(px2)
        odim = int(self.opts["odim"])
        py = np_utils.to_categorical(dy, odim)
        py = np.asarray(py, dtype='float32')
        print("Preprocess for dx1=%s, dx2=%s, dy=%s." % (px1.shape, px2.shape, py.shape))
        return px1, px2, py

    def evaluate(self, predy, goldy):
        assert len(predy) == len(goldy)
        correct = 0
        for p, g in zip(predy, goldy):
            if np.argmax(p) == np.argmax(g):
                correct += 1
        print("Result for classification is %s." % (correct/len(predy)))

    def get_embedding(self):
        opts = self.opts
        if(opts["embed_l"] and opts["embed_w"]):
            try:
                fl = open(opts["embed_l"])
                fw = open(opts["embed_w"])
                # read embeddings
                print("- READING embedding file.")
                the_list = [w.strip() for w in fl]
                the_embed = []
                for i, line in enumerate(fw):
                    the_embed.append([float(n) for n in line.strip().split()])
                if(not len(the_list) == len(the_embed)):
                    print("! Not equal embedding fl and fw.")
                    raise 1
                the_dict = {}
                for word, embed in zip(the_list, the_embed):
                    if(len(embed) != opts["embedding_dims"]):
                        print("! Not equal embedding fw and dim.")
                        raise 1
                    the_dict[word] = embed
                # init
                print("- embedding init.")
                real_dict = opts["dictionary"]
                real_embed = [None for i in range(len(real_dict))]
                for word in real_dict.keys():
                    ind = real_dict[word]
                    if word in the_dict:
                        real_embed[ind] = the_dict[word]
                    elif word.lower() in the_dict:
                        real_embed[ind] = the_dict[word.lower()]
                    else:
                        real_embed[ind] = [np.random.uniform(-1*opts["embed_unk_rand"],1*opts["embed_unk_rand"]) for i in range(opts["embedding_dims"])]
                zz = np.asarray(real_embed, dtype='float32')
                print("- Init Embeding over with ", zz.shape)
                return zz
            except int:
                print("- No embedding init.")
        return None


    def build_untilfinaldense(self, WE=None):   # name must be 'arg1', 'arg2' => 'untilfinaldense'
        # this is the virtual function to be implemented
        raise 1

    # build model only when training
    def build(self):
        opts = self.opts
        print('Build model...')
        self.model = Sequential()
        self.build_untilfinaldense(self.get_embedding())
        # adding the final dense
        odim = int(opts["odim"])
        self.model.add(Dense(odim))
        self.model.add(Activation('softmax'))
        self.model.summary()
        ada = Adagrad(lr=float(opts["lr"]))
        self.model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=["accuracy"])
        # store the arch
        try:
            with open(opts["namep"]+".json", "w") as f:
                f.write(self.model.to_json())
        except:
            print("!! Warning: can not write to json")


    def train_it(self, train_inputs, dev_inputs):
        # 1. load options
        opts = self.opts
        # 2. build model
        self.build()
        # 3. train it
        trainx1, trainx2, trainy = self.preprocess([x[0] for x in train_inputs], [x[1] for x in train_inputs], [x[2] for x in train_inputs])
        devx1, devx2, devy = self.preprocess([x[0] for x in dev_inputs], [x[1] for x in dev_inputs], [x[2] for x in dev_inputs])
        print('Train...')
        c_early = EarlyStopping(patience=3, verbose=1)
        c_saveModel = ModelCheckpoint(opts["namep"]+".hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.model.fit([trainx1, trainx2], trainy, batch_size=opts["batch_size"], nb_epoch=opts["iters"],callbacks=[c_saveModel, c_early],
                       validation_data=([devx1,  devx2], devy))
        print('Train over, and reload best.')
        self.model.load_weights(opts["namep"]+".hdf5")

    def test_it(self, dev_inputs):
        devx1, devx2, devy = self.preprocess([x[0] for x in dev_inputs], [x[1] for x in dev_inputs], [x[2] for x in dev_inputs])
        opts = self.opts
        if self.model is None:
            self.model = model_from_json(opts["namep"]+".json")
            self.model.load_weights(opts["namep"]+".hdf5")
        predy = self.model.predict([devx1, devx2])
        self.evaluate(predy, devy)
        return [np.argmax(i) for i in predy]

class cnn_model(the_model):
    def __init__(self, info={}):
        super(cnn_model, self).__init__(info)

    def build_untilfinaldense(self, WE=None):
        opts = self.opts
        # get options
        max_features = int(opts["max_features"])
        embedding_dims = int(opts["embedding_dims"])
        maxlen = int(opts["maxlen"])
        nb_filter = int(opts["nb_filter"])
        filter_length = int(opts["filter_length"])
        act = opts["activation"]
        pool_length = int(opts["pool_length"])
        hidden_dims = int(opts["hidden_dims"])
        dropout = float(opts["dropout"])
        # start
        model = self.model
        marg1 = Sequential()
        marg2 = Sequential()
        if(not WE is None):
            marg1.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
            marg2.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
        else:
            marg1.add(Embedding(max_features, embedding_dims, input_length=maxlen))
            marg2.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        marg1.add(Dropout(dropout))
        marg2.add(Dropout(dropout))
        marg1.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=act, border_mode='same'))
        marg2.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=act, border_mode='same'))
        marg1.add(MaxPooling1D(pool_length=pool_length))
        marg2.add(MaxPooling1D(pool_length=pool_length))
        marg1.add(Flatten())
        marg2.add(Flatten())
        merged = Merge([marg1, marg2], mode='concat')
        model.add(merged)
        model.add(Dropout(dropout))
        if(hidden_dims > 0):    # whether add another dense
            model.add(Dense(hidden_dims))
            model.add(Dropout(dropout))
        model.add(Activation(act))
        return None

class bow_model(the_model):
    def __init__(self, info={}):
        super(bow_model, self).__init__(info)

    def build_untilfinaldense(self, WE=None):
        opts = self.opts
        # get options
        max_features = int(opts["max_features"])
        embedding_dims = int(opts["embedding_dims"])
        maxlen = int(opts["maxlen"])
        act = opts["activation"]
        hidden_dims = int(opts["hidden_dims"])
        dropout = float(opts["dropout"])
        # start
        model = self.model
        marg1 = Sequential()
        marg2 = Sequential()
        if(not WE is None):
            marg1.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
            marg2.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
        else:
            marg1.add(Embedding(max_features, embedding_dims, input_length=maxlen))
            marg2.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        marg1.add(Dropout(dropout))
        marg2.add(Dropout(dropout))
        marg1.add(Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=(1, embedding_dims)))
        marg2.add(Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=(1, embedding_dims)))
        marg1.add(Flatten())
        marg2.add(Flatten())
        merged = Merge([marg1, marg2], mode='concat')
        model.add(merged)
        model.add(Dropout(dropout))
        if(hidden_dims > 0):    # whether add another dense
            model.add(Dense(hidden_dims))
            model.add(Dropout(dropout))
        model.add(Activation(act))
        return None
