import numpy as np
import cPickle
import sys
import pandas as pd
nameFile = 'spambase.names'
sigmoid = lambda z: 1/(1+np.exp(-z))
def readNames(nameFile, train=True):
    names = []
    print '\tNames generating...'
    with open(nameFile, 'r') as f:
        wr = False
        for line in f:
            line = line.strip()
            if '|' in line:
                continue
            if line == '':
                continue
            line = line.split(':')[0]
            if line == 'word_ratio':
                continue
            if line.startswith('char_'):
                names.append(('x', 'cf', line[-1:]))
            elif line.startswith('capital_'):
                names.append(('x', 'cpt', line[19:]))
            else:
                names.append(('x', 'wr', line))
    if train:
        names.append(('y','',''))
    print '\tDone.'
    return names

def dataParser(filename, feature, scaling = True):
    print 'Data preprocessing...'
    names = readNames(nameFile)
    df = pd.read_csv(filename, index_col=0, header=None, names=names)
    print '\tFeaturing:'
    if feature.startswith('drop:'):
        dropItems = feature[5:].split(',')
        for item in dropItems:
            df = df.drop(item, axis=1, level=1, errors='ignore')
            df = df.drop(item, axis=1, level=2, errors='ignore')

    df['x','cpt'] = np.log(df['x','cpt']+0.0000001)
    df['x','wr'] **= 0.5
    df['x','cf'] **= 0.5
    dataAmt,dimx = df.shape
    print '\tdf:', df.shape
    print '\tfeature done.'
    x = df.values
    x[:,-1] = 1
    y = df.values[:, -1]
    print '\tx:', x.shape, x.dtype, '\n\ty:', y.shape, y.dtype
    if scaling:
        print '\tScaling...'
        mean = df.mean().values[:-1]
        std = df.std().values[:-1]
        x[:,:-1] = (x[:,:-1]-mean)/std
        print '\tDone.'
        scale = {'mean': mean,
                'std':std}
    else:
        scale = {'mean': 0,
                'std': 1 }

    D = np.zeros(dataAmt, dtype=[('x', str(dimx)+'float'), 
                                 ('y', 'float')])
    D['x'] = x
    D['y'] = y
    print '\tD:', D['x'][ -1,-5:], D['y'][-1]
    print 'Done.\n'
    return D, scale

def validate(D, GDpara, foldAmt=4):
    print 'Validating...'
    valScore = [0]*foldAmt
    nDval = len(D)/foldAmt
    nDtrain = len(D) - nDval
    #GDpara['eta'] /= foldAmt
    #GDpara['reg'] /= foldAmt
    print '\tAll Data #', len(D)
    print '\tTraining Data\t#', nDtrain
    print '\tValidation Data\t#', nDval
    for fold in range(foldAmt):
        print 'Fold', fold
        start = len(D)*fold/foldAmt
        Dval = np.zeros(nDval, dtype=D.dtype)
        Dval = D[start : start+nDval]
        Dtrain = np.zeros(nDtrain, dtype=D.dtype)
        Dtrain[:start] = D[:start]
        Dtrain[start:] = D[start+nDval:]
        print '\tValidation Data Range\t:', start, '-', start+nDval-1
        w = LogReg(Dtrain, GDpara)
        print '\nTesting...'
        x,y = Dval['x'], Dval['y']
        diff = y - np.rint( sigmoid(np.dot(x,w)) )
        valScore[fold] = len( diff[diff==0] )/ float(len(diff)) * 100
        print '*'*40
        print '\tFold', fold, 'score: %.4f' % valScore[fold]
        print '*'*40, '\n'
        raw_input()

    return sum(valScore)/foldAmt

def LogReg(D, GDpara):
    print 'Training...'
    x, y = D['x'], D['y']
    nD, dimx = x.shape
    print '\tTraining Data #', nD, ', Dim x:', dimx
    w = np.zeros(dimx, dtype='float')
    itr = GDpara['itr']
    eta = GDpara['eta']
    reg = GDpara['reg']
    G = 0
    if itr>0:
        itr = int(itr)
        for i in range(itr):
            f = sigmoid( np.dot(x,w) )
            g = np.dot(y-f,x) + 2*reg*w
            G += g**2
            w = w+eta*g/(G**0.5)
            diff = y - np.rint(f)
            acc = len( diff[diff==0] )/ float(len(diff)) * 100
            if i%100 == 0:
                print '\t', i, ', acc:', '%.2f' % acc
    else:
        stopRate = -1*itr
        changeRate = 1
        i = 0
        while changeRate > stopRate:
            f = sigmoid( np.dot(x,w) )
            g = np.dot(y-f,x) + 2*reg*w
            G += g**2
            change = eta*g/(G**0.5)
            w = w+change
            changeRate = np.linalg.norm(change)/np.linalg.norm(w)
            diff = y - np.rint(f)
            acc = len( diff[diff==0] )/ float(len(diff)) * 100
            if i%100 == 0:
                print '\t', i, 'change: %.4f' % changeRate, ', acc: %.2f' % acc
            i += 1
        print '\n\t', i-1, 'change: %.4f' % changeRate, ', acc: %.2f' % acc

    print 'Done.\n'
    return w

if __name__ == '__main__':
    feature = 'default'
    GDpara = {'itr':-0.0001,
            'eta':1,
            'reg':0, }
    fileIn = 'spam_data/spam_train.csv'
    fileOut = ''
    val = True
    scaling = True

    for i, arg in enumerate(sys.argv):
        if arg.startswith('-in'):
            fileIn = sys.argv[i+1]
        if arg.startswith('-out'):
            fileOut = sys.argv[i+1]
        if arg.startswith('-feat'):
            # -feat drop:wr,cf
            feature = sys.argv[i+1]
        if arg.startswith('-para:'):
            value = sys.argv[i+1].split(',')
            for i, key in enumerate(arg[6:].split(',')):
                GDpara[key] = float(value[i])
        if arg.startswith('-noVal'):
            val = False
        if arg.startswith('-noScale'):
            scaling = False

    D, scale = dataParser(fileIn, feature, scaling)
    if val:
        valScore = validate(D, GDpara)
    else:
        valScore = 0
    w = LogReg(D, GDpara)

    model = {'w':w,
        'feature': feature,
        'GDpara': GDpara,
        'scale': scale,
        'valScore': valScore, }
    if fileOut == '':
        fileOut = 'val%06.4f_' % valScore
        fileOut += 'dim%02d_' % len(w)
        fileOut += feature
        fileOut += '.m'
    print 'Save as', fileOut, '?'
    raw_input()
    with open(fileOut, 'wb') as f:
        cPickle.dump(model, f)
        print  fileOut, 'saved.'
