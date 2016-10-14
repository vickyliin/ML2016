import cPickle
import numpy as np
import sys
from share import *

def preprocess(filename, trained):
    w = trained['w']
    feature = trained['feature']
    mean = trained['mean']
    std = trained['std']
    print 'Feature:', len(feature)
    x,y = [],[]
    with open(filename,'r') as f:
        data = cPickle.load(f)
        # data[probID, item, hour]
    for i, raw in enumerate(data):
        print 'id', i
        # raw[item, hour]
        d = featureExt(raw,feature)
        if len(mean) == len(data):
            x.append( (d['x']-mean[i])/std[i] )
        else:
            x.append( (d['x']-mean)/std )
        if len(raw) == 10:
            y.append(d['y'])

    x = np.array(x)
    if y == []:
        y = 0
    else:
        y = np.array(y)
    print 'Preprocessed.'
    return x, y

if __name__ == '__main__':
### Open trained results
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        if arg.startswith('-file'):
            wfname = sys.argv[i+1]
    with open(wfname, 'r') as f:
        trained = cPickle.load(f)
        w = trained['w']
        feature = trained['feature']
        mean = trained['mean']
        std = trained['std']

### Find the predict y*
    x,y = preprocess('testData.m', trained)
    if w.ndim == 1:
        y_ = [ sum(w*x[i]) for i in range(len(x))]
    else:
        y_ = [ sum(i) for i in w*x ]
        print w.shape, x.shape
        #raw_input()
    if y != 0:
        meanErr = np.average( np.array(y_)-y )

### Print and save
    if y != 0:
        for i in range(len(y_)):
            print 'id:', str(i), 'y*:', str(y_[i]), 'y:', str(y[i])
        print 'mean error:', meanErr
    else:
        print 'y*', y_
    print wfname[:-2]+'.csv'
    save = raw_input('Save?')
    if save != '1':
        print 'Pass.'
        sys.exit()
    with open(wfname[:-2]+'.csv', 'w') as f:
        f.write('id,value\n')
        for i in range(len(y_)):
            f.write('id_'+str(i)+','+str(y_[i])+'\n')
        print wfname[:-2]+'.csv saved!'
