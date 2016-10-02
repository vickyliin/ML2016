import numpy as np
import cPickle
import sys

trainingDataAmt = 97920
testDataAmt = 43200

def preprocess(fname):
    with open(fname) as f:
        data = cPickle.load(f)
    pm25 = []
    for x in data:
        if x['item'] == 'PM2.5':
            pm25.append(x['value'])
    for i in range( len(pm25)-1 ):
        pm25[i] = [ pm25[i], pm25[i+1] ]
    pm25.pop()
    print 'Preprocessed.'
    return pm25

def GradientLoss(D, w):
    err = sum([ y-(w[0]+w[1]*x) for [x,y] in D ])
    return np.array([ -2*err*w[1], -2*err ])
def Loss(D, w):
    return sum([ pow( y-(w[0]+w[1]*x), 2) for [x,y] in D ])
def err(D, w):
    return sum([ y-(w[0]+w[1]*x) for [x,y] in D ])/len(D)

def GD_Regression(D, eta, itr):
    # model y= w[0] + w[1]x 
    w = np.array([0., 1.]) #initial value
    for i in range(itr):
        w = w-eta*GradientLoss(D, w)
        print w, err(D,w)
    return w

if __name__ == '__main__':
    D = preprocess('trainingData.m')
    eta = np.array([1.0,1.0])
    itr = 100
    for arg in sys.argv:
        if arg.startswith('-itr'):
            itr = int(arg[4:])
        if arg.startswith('-eta'):
            eta = np.array( [ float(arg[4:]) ]*2 )
    w = GD_Regression(D, eta, itr)
    with open('trained_w.m', 'wb') as f:
        cPickle.dump(w,f)
