import numpy as np
import cPickle
import sys

trainingDataAmt = 97920
testDataAmt = 43200

def preprocess(fname):
    with open(fname) as f:
        data = cPickle.load(f)
    pm25 = []
    for i in range( len(data)-1 ):
        if data[i]['item'] == 'PM2.5':
            pm25.append(data[i]['value'])
        if data[i]['month'] != data[i+1]['month']:
            pm25.append(',')
    if data[-1]['item'] == 'PM2.5':
        pm25.append(data[i]['value'])

    D = []
    for i in range( len(pm25)-1 ):
        try:
            D.append( [ pm25[i]*1., pm25[i+1]*1. ] )
        except:
            pass
    print D, 'Preprocessed.'
    return D

def GradientLoss(D, w):
    gradient = np.array([0.,0.])
    for [x,y] in D:
        err = y-(w[0]+w[1]*x)
        gradient[1] += err*x
        gradient[0] += err
    return -2*gradient
def Loss(D, w):
    l = 0
    for [x,y] in D:
        err = y-(w[0]+w[1]*x)
        l += (err*err)
    l /= len(D)
    l = pow(l, 0.5)

    return l

def GD_Regression(D, eta, itr):
    # model y= w[0] + w[1]x 
    w = np.array([0., 1.]) #initial value
    for i in range(itr):
        w = w - eta*GradientLoss(D, w)
        print w, Loss(D,w)
    return w

if __name__ == '__main__':
    D = preprocess('trainingData.m')
    eta = np.array([0.00000023]*2)
    itr = 100
    for arg in sys.argv:
        if arg.startswith('-itr'):
            itr = int(arg[4:])
        if arg.startswith('-eta'):
            eta = np.array( [ float(arg[4:]) ]*2 )
    w = GD_Regression(D, eta, itr)
    with open('trained_w.m', 'wb') as f:
        cPickle.dump(w,f)
