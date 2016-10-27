import pandas as pd
import numpy as np
import cPickle
import sys
from pprint import pprint
nameFile = 'spambase.names'
sigmoid = lambda z: 1/(1+np.exp(-1*z))
from train import readNames

def dataParser(filename, model):
    print 'Data preprocessing...'
    names = readNames(nameFile, train=False)
    df = pd.read_csv(filename, index_col=0, header=None, names=names)
    print '\tFeaturing:'
    feature = model['feature']
    if feature.startswith('drop:'):
        dropItems = feature[5:].split(',')
        for item in dropItems:
            df = df.drop(item, axis=1, level=1, errors='ignore')
            df = df.drop(item, axis=1, level=2, errors='ignore')

    df['x','cpt'] = np.log(df['x','cpt']+0.0000001)
    df['x','wr'] **= 0.5
    df['x','cf'] **= 0.5

    dataAmt,dimx = df.shape
    dimx += 1
    print '\tdf:', df.shape
    x = np.ones( dataAmt, dtype='%dfloat' % dimx)
    x[:,:-1] = df.values
    print '\tx:', x.shape, x.dtype

    print '\tScaling:'
    mean = model['scale']['mean']
    std = model['scale']['std']
    x[:,:-1] = (x[:,:-1]-mean)/std

    print 'Done.\n'
    return x



if __name__ == '__main__':
    fileModel = 'LogReg.m'
    fileTest = 'spam_data/spam_test.csv'
    fileOut = ''
    for i, arg in enumerate(sys.argv):
        if arg.startswith('-model'):
            fileModel = sys.argv[i+1]
        if arg.startswith('-out'):
            fileOut = sys.argv[i+1]
        if arg.startswith('-test'):
            fileTest = sys.argv[i+1]

    with open(fileModel, 'r') as f:
        model = cPickle.load(f)
        print fileModel, 'loaded'
        print 'Feature:', model['feature']
        print 'valScore:', model['valScore']
    raw_input()
    
    x = dataParser(fileTest, model)
    w = model['w']
    yGuess = np.rint( sigmoid(np.dot(x,w)) )
    if fileOut == '':
        fileOut = fileModel[:-2] + '.csv'
    print 'Save as', fileOut, '?'
    raw_input()
    with open(fileOut, 'w') as f:
        f.write('id,label\n')
        for i,y in enumerate(yGuess):
            f.write('%d,%d\n' % (i+1, y))
    print fileOut, 'Saved.'
