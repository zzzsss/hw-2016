import numpy as np
from keras.models import Sequential
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
    DEFAULT_OPS = {"max_features": None, "embedding_dims": 100, "maxlen": 60, "nb_filter": 256, "filter_length": 3, "activation": 'tanh', "pool_length": 60, "hidden_dims": 128,  "batch_size": 128, "iters": 10, "lr": 0.01, "dropout": 0.25,
                   "embed_l": None, "embed_w": None, "embed_unk_rand": 0.1, "dictionary": None,
                   "odim": 5,
                   "namep": "sst"}

    def __init__(self, info={}):
        self.model = None
        self.opts = the_model.DEFAULT_OPS
        self.configs = dict()
        opts = self.opts
        for k in info:
            opts[k] = info[k]
        print("-- Final options are:")
        print(self.opts)

    def preprocess(self, dx, dy):
        print("Preprocess data.")
        px = sequence.pad_sequences(dx, maxlen=self.opts["maxlen"])
        px = np.asarray(px)
        odim = int(self.opts["odim"])
        py = None
        if(odim == 1):
            py = np.asarray(dy, dtype='float32')
        elif(odim > 1):    # [0,1] -> category
            pcat = [int(s/(1.001/odim)) for s in dy]  # get rid of 0 and 1
            py = np_utils.to_categorical(pcat, odim)
            py = np.asarray(py, dtype='float32')
        print("Preprocess for dx=%s, dy=%s." % (px.shape, py.shape))
        return px, py

    def evaluate(self, predy, scorey):
        assert len(predy) == len(scorey)
        odim = int(self.opts["odim"])
        if(odim == 1):
            err = 0.
            correct2 = 0
            correct5 = 0
            for p, s in zip(predy, scorey):
                err += (p-s)**2
                if int(s/(1.001/2)) == int(p/(1.001/2)):
                    correct2 += 1
                if int(s/(1.001/5)) == int(p/(1.001/5)):
                    correct5 += 1
            err, correct2, correct5 = err/len(predy), correct2/len(predy), correct5/len(predy)
            print("Result for regression is %s, %s and %s." % (np.sqrt(err), correct2, correct5))
        elif(odim > 1):
            correct = 0
            for p, s in zip(predy, scorey):
                if int(s/(1.001/odim)) == np.argmax(p):
                    correct += 1
            print("Result for classification %d is %s." % (odim, correct/len(predy)))
        else:
            raise 1

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


    def build_untilfinaldense(self, WE=None):
        # this is the virtual function to be implemented
        raise 1

    # build model only when training
    def build(self):
        opts = self.opts
        print('Build model...')
        self.model = self.build_untilfinaldense(self.get_embedding())
        # adding the final dense
        odim = int(opts["odim"])
        self.model.add(Dense(odim))
        if(odim > 1):
            self.model.add(Activation('softmax'))
        self.model.summary()
        ada = Adagrad(lr=float(opts["lr"]))
        if(odim == 1):
            self.model.compile(loss='mean_squared_error', optimizer=ada)
        elif(odim > 1):
            self.model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=["accuracy"])
        else:
            raise 1
        # store the arch
        try:
            with open(opts["namep"]+".json", "w") as f:
                f.write(self.model.to_json())
        except:
            print("!! Warning: can not write to json")


    def train_it(self, trainx, trainy, devx, devy):
        # 1. load options
        opts = self.opts
        # 2. build model
        self.build()
        # 3. train it
        trainx, trainy = self.preprocess(trainx, trainy)
        devx, devy = self.preprocess(devx, devy)
        print('Train...')
        c_early = EarlyStopping(patience=3, verbose=1)
        which_monitor = 'val_acc'
        if(int(opts["odim"]) == 1):
            which_monitor = 'val_loss'
        c_saveModel = ModelCheckpoint(opts["namep"]+".hdf5", monitor=which_monitor, verbose=1, save_best_only=True, mode='auto')
        self.model.fit(trainx, trainy, batch_size=opts["batch_size"], nb_epoch=opts["iters"],callbacks=[c_saveModel, c_early],
                       validation_data=(devx, devy))
        print('Train over, and reload best.')
        self.model.load_weights(opts["namep"]+".hdf5")

    def test_it(self, testx, testy):
        scorey = testy
        opts = self.opts
        if self.model is None:
            self.model = model_from_json(opts["namep"]+".json")
            self.model.load_weights(opts["namep"]+".hdf5")
        testx, testy = self.preprocess(testx, testy)
        predy = self.model.predict(testx)
        self.evaluate(predy, scorey)


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
        model = Sequential()
        if(not WE is None):
            model.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
        else:
            model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(dropout))
        model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=act, border_mode='same'))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(Flatten())
        model.add(Dropout(dropout))
        if(hidden_dims > 0):    # whether add another dense
            model.add(Dense(hidden_dims))
            model.add(Dropout(dropout))
        model.add(Activation(act))
        return model

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
        model = Sequential()
        if(not WE is None):
            model.add(Embedding(max_features, embedding_dims, input_length=maxlen, weights=[WE]))
        else:
            model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(dropout))
        model.add(Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=(1, embedding_dims)))     # simple average
        model.add(Flatten())
        model.add(Dropout(dropout))
        if(hidden_dims > 0):    # whether add another dense
            model.add(Dense(hidden_dims))
            model.add(Dropout(dropout))
        model.add(Activation(act))
        return model



