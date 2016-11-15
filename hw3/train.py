from util import *
import json
import cPickle
import numpy as np
import sys
import keras.backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

'''
class YModel(Sequential):
    def __init__(self, img_channels=3, img_rows=32,
                 img_cols=32, nb_classes=10):
        super(YModel, self).__init__()
        self.add(Convolution2D(64, 5, 5, dim_ordering='th',
                                input_shape=(img_channels, img_rows, img_cols)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.65))

        self.add(Convolution2D(128, 3, 3, dim_ordering='th'))
        self.add(Activation('relu'))
        self.add(AveragePooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.65))

        self.add(Convolution2D(192, 3, 3, dim_ordering='th'))
        self.add(Activation('relu'))
        self.add(AveragePooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.65))

        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('tanh'))
        self.add(Dropout(0.65))
        self.add(Dense(nb_classes))
        self.add(Activation('softmax'))
        adam = Adam(lr=5e-4)
        self.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'],
                      )
        self.summary()
'''
def createModel(dim_ordering='th',
                input_shape=(3,32,32),
                nb_classes=10, ):
    
    model = Sequential()
    model.add(Convolution2D(15, 5,5, dim_ordering=dim_ordering, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering=dim_ordering))
    model.add(Dropout(0.65))

    model.add(Convolution2D(20, 3,3, dim_ordering=dim_ordering))
    model.add(Activation('relu'))
    model.add(Dropout(0.65))
    model.add(Convolution2D(300, 3,3, dim_ordering=dim_ordering))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering=dim_ordering))
    model.add(Dropout(0.65))
    '''
    model.add(Convolution2D(50, 3,3, border_mode='same', dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
    '''
    model.add(Flatten())
    model.add(Dropout(0.65))
    model.add(Dense(4*nb_classes))
    model.add(Activation('relu'))
    model.add(Dense(2*nb_classes))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('relu'))
    model.add(Activation('softmax'))

    adam = Adam(lr=1e-3)

    model.compile(loss='categorical_crossentropy', optimizer=adam, init='lecun_uniform', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    path = 'data'
    modelOut = 'trained_model'

    for i,arg in enumerate(sys.argv):
        if arg.startswith('-dir'):
            path = sys.argv[i+1]
            if path[-1] == '/':
                path = path[:-1]
        if arg.startswith('-save'):
            modelOut = sys.argv[i+1]

    # load/normalize/reordering data
    Xl, Xu, Xt = load_data(path, data=['l'])
    Xl = np.swapaxes(Xl.reshape(10,500,3072), 0,1).reshape(5000,3,32,32)
    Yl = np.array(list([np.identity(10)])*500).reshape(5000,10)

    # create/train/save model with labeled data
    print 'Creating model with labeled data...'
    model = createModel()
    model.summary()
    raw_input('...')
    earlyStop = EarlyStopping(monitor='val_loss', patience=50)
    checkpointer = ModelCheckpoint(modelOut)
    histl = model.fit( Xl, Yl, 
            batch_size=300, 
            nb_epoch=1000, 
            validation_split=0.2, 
            callbacks=[earlyStop, checkpointer], )
    print 'The model has been trained and saved as %s!\n' % modelOut

    # save the history
    print 'Saving the history...'
    histl = json.dumps(histl.history, indent=4)+'\n'
    with open('hist_%s' % modelOut, 'w') as f:
        f.write(histl)
    print 'hist_%s saved!' % modelOut

