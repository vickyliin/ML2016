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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

nb_classes=20
encoded_dim=100
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

    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_model.get_weights())
 

## Test Function ##
def encode(F, encoder):
    X = F.toarray()
    Xenc = encoder.predict(X)
    return Xenc.reshape(-1, encoded_dim)
def predict(Xtenc, model):
    Ypred = model.predict(Xtenc)
    return Ypred

## Training Autoencoder ##
def encoderGen(F, nb_epoch=1000, dimOut=100):
    print('Model generating...')
    F = F.toarray()
    dimIn = F[0].shape[0]

    sentence = Input(shape=(dimIn,))

    encoded = Dense(dimOut, activation='sigmoid')(sentence)
    encoder = Model(input=sentence, output=encoded)
    print('Encoder generated!')

    decoded = Dense(dimIn, activation='sigmoid')(encoded)
    autoencoder = Model(sentence, decoded)
    autoencoder.summary()
    print('Autoencoder generated!')

    nadam = Nadam(lr=0.008)
    print('Compiling...')
    autoencoder.compile(optimizer=nadam, 
                        loss='binary_crossentropy', 
                        )
    csvLogger = CSVLogger('autoencoder.log')
    monitor = 'val_loss'
    earlyStop = EarlyStopping(monitor=monitor, 
                              patience=30)
    checkPointer = saveBestModel(monitor=monitor) 

    print('Preparing to train...')
    autoencoder.fit(F, F,
                    nb_epoch=nb_epoch,
                    batch_size=128,
                    validation_split=0.2, 
                    callbacks=[earlyStop, csvLogger, checkPointer],
                    )
    len_encoder = len(encoder.get_weights())
    return encoder

