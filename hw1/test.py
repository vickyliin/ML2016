import cPickle
import numpy as np
import sys
from train import featureApply

def preprocess(filename, feature):
    print 'Feature:', feature
    x,y = [],[]
    with open(filename,'r') as f:
        data = cPickle.load(f)
        # data[probID, item, hour]
    for raw in data:
        d = featureApply(raw,feature)
        x.append(d['x'])
        if len(raw) == 10:
            y.append(d['y'])

    x = np.array(x)
    if y == []:
        y = 0
    else:
        y = np.array(y)
    raw_input('Preprocessed.')
    return x, y

if __name__ == '__main__':
### Open trained results
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        if arg.startswith('-file'):
            wfname = sys.argv[i+1]
    with open(wfname+'.w', 'r') as f:
        trained = cPickle.load(f)
        w = trained['w']
        feature = trained['feature']

### Find the predict y*
    x,y = preprocess('testData.m', feature)
    if w.ndim == 1:
        y_ = [ sum(w*x[i]) for i in range(len(x))]
    else:
        y_ = [ sum(i) for i in w*x ]
    if y != 0:
        meanErr = np.average( np.array(y_)-y )

### Print ans save
    if y != 0:
        for i in range(len(y_)):
            print 'id:', str(i), 'y*:', str(y_[i]), 'y:', str(y[i])
        print 'mean error:', meanErr
    else:
        for i in range(len(y_)):
            print 'id:', str(i), 'y*:', str(y_[i])
    save = raw_input('Save?')
    if save != '1':
        print 'Pass.'
        sys.exit()
    with open(wfname+'.csv', 'w') as f:
        f.write('id,value\n')
        for i in range(len(y_)):
            f.write('id_'+str(i)+','+str(y_[i])+'\n')
        print wfname+'.csv saved!'
