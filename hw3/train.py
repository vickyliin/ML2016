from util import *
import cPickle
import numpy as np
import sys
import keras.backend
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD, Adam
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

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

class myModel(Sequential):
    def __init__(self, dim_ordering='th',
                 input_shape=(3,32,32),
                 nb_classes=10, 
                ):
        super(myModel, self).__init__()
        self.add(Convolution2D(15, 5,5, dim_ordering=dim_ordering, input_shape=input_shape))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2,2), dim_ordering=dim_ordering))
        self.add(Dropout(0.65))

        self.add(Convolution2D(20, 3,3, dim_ordering=dim_ordering))
        self.add(Activation('relu'))
        self.add(Dropout(0.65))
        self.add(Convolution2D(300, 3,3, dim_ordering=dim_ordering))
        self.add(Activation('relu'))

        self.add(MaxPooling2D(pool_size=(2,2), dim_ordering=dim_ordering))
        self.add(Dropout(0.65))
        '''
        self.add(Convolution2D(50, 3,3, border_mode='same', dim_ordering='th'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
        '''

        #self.add(Dropout(0.25))

        
        self.add(Flatten())
        self.add(Dropout(0.65))
        self.add(Dense(4*nb_classes))
        self.add(Activation('relu'))
        self.add(Dense(2*nb_classes))
        self.add(Activation('relu'))
        self.add(Dense(nb_classes))
        self.add(Activation('relu'))
        self.add(Activation('softmax'))

        adam = Adam(lr=1e-3)

        self.compile(loss='categorical_crossentropy', optimizer=adam, init='lecun_uniform', metrics=['accuracy'])
        self.summary()

if __name__ == '__main__':
    path = 'data'
    output_model = 'trained_model'
    input_model = output_model
    test = False

    for i,arg in enumerate(sys.argv):
        if arg.startswith('-dir'):
            path = sys.argv[i+1]
            if path[-1] == '/':
                path = path[:-1]
        if arg.startswith('-save'):
            output_model = sys.argv[i+1]
        if arg.startswith('-test'):
            test = True
        if arg.startswith('-load'):
            input_model = sys.argv[i+1]

    # load/normalize/reordering data
    Xl, Xu, Xt = load_data(path, data=['l'])
    Xl = np.swapaxes(Xl.reshape(10,500,3072), 0,1).reshape(5000,3,32,32)
    Yl = np.array(list([np.identity(10)])*500).reshape(5000,10)

    # train model
    model = myModel()
    raw_input()
    model.fit(Xl, Yl, batch_size=300, nb_epoch=1000, validation_split=0.2)
    model.save(output_model)

