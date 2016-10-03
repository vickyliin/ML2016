import numpy as np
import cPickle
import sys

trainingDataAmt = 97920
testDataAmt = 43200
D_size = 5652               # 240*24-9*12
items = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2',
        'NOx','O3','PM10','PM2.5','RH','SO2','THC',
        'WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
itemIdx = {}
i = 1
for item in items:
    itemIdx[item] = i
    i += 1
print itemIdx


def preprocess(fname, decayRate):
    decay = np.exp( np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,0])*decayRate )
    decay /= sum(decay)
    raw_input(decay)
    with open(fname) as f:
        data = cPickle.load(f)
    
    seqs = {} # seqs[item][i]: the ith month's value sequence of item
    for item in items:
        seqs[item] = [ [] for i in range(12) ]
    for d in data:
        seqs[ d['item'] ][ d['month']-1 ].append( d['value'] )

    D = np.zeros(D_size, dtype=[ ('x', '18float') , ('y','float') ])
    for item in items:
        i = 0
        for seq in seqs[item]:
            for j in range( len(seq)-9 ):
                D[i]['x'][ itemIdx[item] ] = sum( [decay[k]*seq[j+k] for k in range(9)] )
                if item == 'PM2.5':
                    D[i]['y'] = seq[j+9]
                    D[i]['x'][0] = 1
                i += 1
    print D, 'Preprocessed.'
    return D

def GradientLoss(D, w, rate):
    gradient = np.zeros(len(w), dtype='float')
    for i in range( len(w) ):
        for (x,y) in D:
            err = y-sum(w*x)
            gradient[i] += err*x[i]
        gradient[i] *= -2
        gradient[i] += rate*2*w[i]
    return gradient
def Loss(D, w):
    l = 0
    for (x,y) in D:
        err = y-sum(w*x)
        l += err*err
    l /= len(D)
    l = pow(l, 0.5)
    return l
def reg(w):
    reg = 0
    for x in w:
        reg += x*x
    return reg

def GD_Regression(D, eta, itr, regRate, init=1):
    # model y= sum_item(w[item]x_item) 
    w = np.zeros( len(items)+1, dtype='float' )
    w[ itemIdx['PM2.5'] ] = init #initial value
    for i in range(itr):
        w = w-eta*GradientLoss(D, w, regRate)
        print w, Loss(D,w), reg(w)
    return w

if __name__ == '__main__':
    eta = np.array( [0.000000006]*(len(items)+1) )
    itr = 100
    decayRate = 1
    regRate = 0
    init = 1 # initial PM2.5 weight
    wfname = sys.argv[0][:-9]
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        wfname += arg
        if arg.startswith('-itemIdx'):
            with open(wfname, 'wb') as f:
                cPickle.dump(itemIdx, f)
                print wfname+' saved!'
            sys.exit()
        if arg.startswith('-itr'):
            itr = int(arg[4:])
        if arg.startswith('-eta'):
            eta = np.array( [ float(arg[4:]) ]*(len(items)+1) )
        if arg.startswith('-decay'):
            decayRate = float(arg[6:])
        if arg.startswith('-reg'):
            regRate = float(arg[4:])
        if arg.startswith('-init'):
            init = float(arg[5:])
    wfname += '.w'
    D = preprocess('trainingData.m', decayRate)
    w = GD_Regression(D, eta, itr, regRate, init)
    with open(wfname, 'wb') as f:
        cPickle.dump(w,f)
        print wfname+' saved!'
