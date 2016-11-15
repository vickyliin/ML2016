from util import *
import cPickle
import numpy as np
import sys
import keras.backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)
nb_classes=10

if __name__ == '__main__':
    path = 'data'
    csvOut = 'prediction.csv'
    modelIn = 'trained_model'

    for i,arg in enumerate(sys.argv):
        if arg.startswith('-dir'):
            path = sys.argv[i+1]
            if path[-1] == '/':
                path = path[:-1]
        if arg.startswith('-save'):
            csvOut = sys.argv[i+1]
        if arg.startswith('-load'):
            modelIn = sys.argv[i+1]
        if arg.startswith('-test'):
            nb_epoch = 5
            modelIn = 'test_trained_model'

    # load/normalize/reordering data
    Xl, Xu, Xt = load_data(path, data=['t'])

    # load model
    print 'Loading model...'
    model = load_model(modelIn)
    model.summary()
    print 'Model Loaded!'

    # predict/write
    print 'Predicting the classes...'
    Ct = model.predict_classes(Xt)
    print 'Done!'
    raw_input( 'Save as %s?' % csvOut )
    with open(csvOut, 'w') as f:
        f.write('ID,class\n')
        for i, c in enumerate(Ct):
            f.write('%d,%d\n' % (i,c))
    print '%s saved!' % csvOut

