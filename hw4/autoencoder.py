import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix

from time import time
import json , cPickle , sys 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

nb_classes=10
encoded_dim=256
best_weights=[]

class saveBestModel(ModelCheckpoint):
    def __init__(self, monitor='val_loss',
                 mode='auto'):
        super(saveBestModel, self).__init__(
                filepath=None, 
                monitor=monitor, 
                mode=mode)
        self.best_model = None
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        elif self.monitor_op(current, self.best):
            self.best = current
            self.best_model = self.model
            print '*',

    def on_train_end(self, logs={}):
        global best_weights
        best_weights = self.best_model.get_weights()
 

## Load ##
def load(folder='data/', data=['l', 'u', 't']):
    name = {'l':'all_label.p', 
            'u':'all_unlabel.p',
            't':'test.p', }
    X = {}
    if folder[-1] != '/':
        folder += '/'
    for d in data:
        path = folder + name[d]
        print 'Loading %s...' % name[d]
        with open(path, 'rb') as f:
            X[d] = cPickle.load(f)
        print name[d], 'loaded\n'
    
    print 'Preprocessing...'
    X['l'] = np.array(X['l'], dtype='float64').swapaxes(0,1).reshape(-1,3072)
    X['u'] = np.array(X['u'], dtype='float64')
    X['t'] = np.array(X['t']['data'], dtype='float64')
    for d in data:
        print name[d], '\n\tdtype:\t', X[d].dtype
        print '\tshape:\tfrom ', X[d].shape,
        X[d] = X[d].reshape(-1,3,1024).swapaxes(1,2).reshape(-1,32,32,3)
        print 'to', X[d].shape
        X[d] /= 255
        print '\tRange:\t[%.2f, %.2f]\n' % (np.min(X[d]),
                                                   np.max(X[d]))
    return X

## Test Function ##
def encode(X, encoder):
    Xenc = encoder.predict(X)
    return Xenc.reshape(-1, encoded_dim)
def predict(Xtenc, model):
    Ypred = model.predict(Xtenc)
    return Ypred

def save(csvOut, Ypred):
    with open(csvOut, 'w') as f:
        f.write('ID,class\n')
        for i, c in enumerate(Ypred):
            f.write('%d,%d\n' % (i,c))
    print '%s saved!' % csvOut


## Training Autoencoder ##
def encoderGen(x, nb_epoch=1000):
    if type(x) == dict:
        x = np.array( sum([ list(x[d]) for d in x ], []) )
    np.random.shuffle(x)
    x_train = x.copy()
    print x.shape

    print 'Model generating...'
    input_img = Input(shape=x_train[0].shape)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoded_dim, activation='sigmoid')(x)
    encoder = Model(input=input_img, output=encoded)
    print 'Encoder generated!'
    length = encoder.layers[-2].output_shape[1]
    shape = encoder.layers[-2].input_shape[1:]
    print length, shape
    raw_input('...')

    x = Dense(length, activation='relu')(encoded)
    x = Reshape(shape)(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    print 'Autoencoder generated!'

    nadam = Nadam(lr=0.008)
    print 'Compiling...'
    autoencoder.compile(optimizer=nadam,  
                        loss='binary_crossentropy',
                        )
    csvLogger = CSVLogger('autoencoder.log')
    monitor = 'val_loss'
    earlyStop = EarlyStopping(monitor=monitor, 
                              patience=30)
    checkPointer = saveBestModel(monitor=monitor) 

    print 'Preparing to train...'
    autoencoder.fit(x_train, x_train,
                    nb_epoch=nb_epoch,
                    batch_size=128,
                    validation_split=0.2, 
                    callbacks=[earlyStop, csvLogger, checkPointer],
                    )
    len_encoder = len(encoder.get_weights())
    encoder.set_weights(best_weights[:len_encoder])
    return encoder

## Training Propogation Model
def train(Xl, Xu, nb_val=100, nb_train=1000, 
          gamma=20, kernel='knn',
          max_iter=5, tol=0.01):
    # Xl/Xu : <<Encoded>> label/unlabel data
    print 'Shuffle and split the validation set'
    Yl = range(nb_classes) * (len(Xl)/nb_classes)
    Yu = [-1] * len(Xu)
    shuff = zip(Xl,Yl)
    np.random.shuffle(shuff)
    Xval, Yval = zip(*shuff[:nb_val])
    Xl, Yl = zip(*shuff[nb_val:])

    Xlu = list(Xl)+list(Xu)
    Ylu = list(Yl)+Yu
    shuff = zip(Xlu,Ylu)
    np.random.shuffle(shuff)
    Xlu, Ylu = zip(*shuff)

    Xlu = np.array(Xlu)[:nb_train]
    Ylu = np.array(Ylu)[:nb_train]
    Xval = np.array(Xval)
    Yval = np.array(Yval)

    print 'Create Model'
    propModel = LabelSpreading(
            kernel=kernel, 
            gamma=gamma, 
            max_iter=max_iter,
            tol=tol)
    print 'Training...'
    start = time()
    propModel.fit(Xlu, Ylu)
    Ypred = propModel.predict(Xval)
    end = time()
    print 'Training time: %.2f secs' % (end-start)
    print 'Score:', propModel.score(Xval, Yval)
    print 'Confusion Matrix:'
    print confusion_matrix(Yval, Ypred)
    return propModel
