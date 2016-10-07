import numpy as np
import cPickle
import sys

trainingDataAmt = 97920
testDataAmt = 43200
testProbAmt = 240
D_size = 5652               # 240*24-9*12
D_size_flt = 180            # 20*9
itemIdx = {}

def preprocess(fname, decayRate, feature, flt):
    dimx = len(feature)+1
    i = 1
    for item in feature:
        itemIdx[item] = i
        i += 1
    decay = np.exp( np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,0])*decayRate )
    decay /= sum(decay)
    raw_input(decay)
    print feature

    with open(fname) as f:
        data = cPickle.load(f)

    if flt:
        with open('mostSim.m','r') as f:
            mostSim = cPickle.load(f)
        fltSet = zip([mostSim])
        D = np.zeros([testProbAmt, D_size_flt], dtype=[ ('x', str(dimx)+'float') , ('y','float') ])
        # D[i, {(x,y)} ]: the training data set of i'th test data
        for i in range(testProbAmt):
            for d in data:
                if d['item'] in feature and (d['month'],d['day'],d['hour']) in fltSet[i]:
                    pass
    else:
        seqs = np.zeros([len(feature),12,480], 
                dtype=[('day','int'), ('hour', 'int'), ('value', 'float')])
        # seqs[i][j]: item i, month j sequence of value/day/hour
        for d in data:
            if d['item'] in feature:
                seqs[ itemIdx[d['item']] ][ d['month']-1 ][i] = append( d['value'] )

        D = np.zeros(D_size, dtype=[ ('x', str(dimx)+'float') , ('y','float') ])
        # x_0 remain for constant term (1)
        for item in feature:
            i = 0
            for seq in seqs[item]:
                for j in range( len(seq)-9 ):
                    D[i]['x'][ itemIdx[item] ] = sum( [decay[k]*seq[j+k] for k in range(9)] )
                    if item == 'PM2.5':
                        D[i]['y'] = seq[j+9]
                        D[i]['x'][0] = 1
                    i += 1
        print 'Preprocessed.'
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

def LA_Regression(D):
    dimx = len(D[0]['x'])
    A = np.matrix(D['x'])
    b = np.matrix(D['y']).T
    w = np.matrix(np.zeros(dimx, dtype='float'))
    w = (A.T*A).I*A.T*b
    print w.T
    return np.array(w.T)[0]
    
def GD_Regression(D, regRate, GDpara):
    # model y= sum_item(w[item]x_item)
    dimx = len(D[0]['x'])
    eta,itr,init = GDpara[0],GDpara[1],GDpara[2]

    w = np.zeros( dimx, dtype='float' )
    G = np.zeros( dimx, dtype='float' )
    w[ itemIdx['PM2.5'] ] = init #initial value
    if itr>0:
        for i in range(itr):
            g = GradientLoss(D, w, regRate)
            G += g*g
            w = w-eta*g/np.power(G,0.5)
            print i, w, Loss(D,w), reg(w)
    else:
        stopRate = -1*itr
        w_last = np.zeros(dimx, dtype='float')
        changeRate = np.ones(dimx, dtype='float')
        while abs(np.average(changeRate)) >= stopRate:
            g = GradientLoss(D, w, regRate)
            print g
            G += g*g
            changeRate = eta*g/np.power(G,0.5)
            w = w-changeRate
            changeRate = changeRate/w
            print np.average(changeRate), w, Loss(D,w), reg(w)
    return w

if __name__ == '__main__':
    itr = 100
    eta = [1]
    decayRate = 1
    regRate = 0
    init = 1 # initial PM2.5 weight
    wfname = sys.argv[0][:-9]
    la = False
    flt = False
    feature = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2',
            'NOx','O3','PM10','PM2.5','RH','SO2','THC',
            'WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
    for i in range( 1, len(sys.argv) ):
        arg = sys.argv[i]
        wfname += arg
        if arg.startswith('-itr'):
            itr = float(arg[4:])
        if arg.startswith('-eta'):
            eta = [float(arg[4:])]
        if arg.startswith('-decay'):
            decayRate = float(arg[6:])
        if arg.startswith('-reg'):
            if arg == '-reg':
                regRate = -1
            regRate = float(arg[4:])
        if arg.startswith('-init'):
            init = float(arg[5:])
        if arg.startswith('-open'):
            feature = arg[5:].split(',')
        if arg.startswith('-close'):
            for item in arg[6:].split(','):
                feature.remove(item)
        if arg == '-la':
            la = True
        if arg == '-flt':
            flt = True
    wfname += '.w'
    eta = np.array( eta*(len(feature)+1) )
    D = preprocess('trainingData.m', decayRate, feature, flt)
    if la:
        w = LA_Regression(D)
    else:
        w = GD_Regression(D, regRate, [eta, itr, init])
        if regRate == 0:
            print LA_Regression(D)
    save = raw_input('Save?')
    if save == '1':
        with open(wfname, 'wb') as f:
            cPickle.dump(w,f)
            print wfname+' saved!'
