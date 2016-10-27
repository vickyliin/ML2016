import numpy as np
import cPickle
import sys
import pandas as pd
nameFile = 'spambase.names'
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

def dataParser(filename, feature):
    print 'Data preprocessing...'
    names = readNames(nameFile)
    df = pd.read_csv(filename, index_col=0, header=None, names=names)
    print '\tFeaturing:'
    if feature.startswith('drop:'):
        dropItems = feature[5:].split(',')
        for item in dropItems:
            df = df.drop(item, axis=1, level=1, errors='ignore')
            df = df.drop(item, axis=1, level=2, errors='ignore')
    print '\tdf:', df.shape
    raw_input('Done.')
    
    return df

def classify(x, tree):
    node = tree[0]
    while type(node) != int:
        attr = node['attr']
        c = node['c']
        if x[ node['attr'] ] < c:
            node = tree[ node['son'][0] ]
            print('l'),
        else:
            node = tree[ node['son'][1] ]
            print('r'),
    print ''
    return node

def validate(D, stopUnity, foldAmt=4):
    print 'Validating...'
    valScore = [0.0]*foldAmt
    nDval = len(D)/foldAmt
    nDtrain = len(D) - nDval
    print '\tAll Data #', len(D)
    print '\tTraining Data\t#', nDtrain
    print '\tValidation Data\t#', nDval
    nameDic = {}
    for i, name in enumerate(D): 
        nameDic[name] = i
    for fold in range(foldAmt):
        print 'Fold', fold
        start = len(D)*fold/foldAmt
        Dval = D[start : start+nDval]
        Dtrain = pd.DataFrame( np.zeros([nDtrain, D.shape[1]], dtype=D.values.dtype) )
        print Dtrain.shape
        print D[:start].values.shape
        Dtrain[:start] = D[:start].values
        Dtrain[start:] = D[start+nDval:].values
        print '\tValidation Data Range\t:', start, '-', start+nDval-1

        tree = [0]
        attrs = list(D.columns)
        del(attrs[-1])
        trainTree(tree, 0, D, attrs, stopUnity)
        print '\n\tTree nodes:', len(tree)
        print '\tFold %d trained.' % fold
        
        print '\nTesting...'
        for i, node in enumerate(tree):
            if type(node) != int:
                tree[i]['attr'] = nameDic[ node['attr'] ]
        for x in Dval.values:
            if classify(x, tree) == x[-1]:
                valScore[fold] += 1
        valScore[fold] /= (len(Dval)/100)

        print '*'*40
        print '\tFold', fold, 'score: %.4f' % valScore[fold]
        print '*'*40, '\n'
        raw_input()

    return sum(valScore)/foldAmt

def findAttr(D, attrs):
    splitD = {}
    classCnt = np.array( [ 0, D['y'].sum() ] )
    classCnt[0] = D.shape[0] - classCnt[1]

    minH = len(D)
    for attr in attrs:
        # lCnt[j] = number of which y=j before i
        # rCnt[j] = number of which y=j after i
        lCnt = np.zeros(2, dtype='int')
        sortY = D.sort_values(by=attr)['y'].values

        lCnt[ sortY[0] ] += 1
        for i in range( 1, len(D) ):
            # i: total data amout at the left
            rCnt = classCnt - lCnt
            weightH = sum( lCnt*np.log(lCnt+0.0001) ) + sum(rCnt*np.log(rCnt+0.0001) ) - ( 
                    i*np.log(i+0.0001) + (len(D)-i)*np.log(len(D)-i+0.0001) )
            weightH = -weightH
            #print '\t', attr, i, 'lCnt:', lCnt, 'weightH:', weightH
            if weightH < minH:
                minH = weightH
                cut = i
                attrS = attr

            lCnt[ sortY[i] ] += 1
        print '\t', 'attrS:', attrS, 'cut:', cut, 'minH:', minH

    D = D.sort_values(by=attrS)
    splitD={ 'left': D[:cut],
            'right': D[cut:]}
    c = splitD['right'][attrS].values[0]

    print '\n\tsplitD:', splitD['left'].shape[0], splitD['right'].shape[0]
    print '\tattrS:', attrS, 'c:', c
    #raw_input()
    return attrS, c, splitD

def trainTree(tree, idx, D, attrs, stopUnity=0.9):
    unity1 = len ( D[ D['y']==1 ] ) / float( len(D) )
    print '[Y/N] Ratio %.2f : %.2f' % (unity1*100, (1-unity1)*100 )
    if unity1 > stopUnity:
        tree[idx] = 1
        print 'Reach leaf node 1 with %d data' % len(D)
    elif ( 1-unity1 ) > stopUnity:
        tree[idx] = 0
        print 'Reach leaf node 0 with %d data' % len(D)

    else:
        attr, c, splitD = findAttr(D, attrs)

        lidx, ridx = len(tree), len(tree)+1
        tree.append({}), tree.append({})

        node = { 'attr': attr,
                'c': c,
                'son': [lidx, ridx], }
        tree[idx] = node

        trainTree(tree, lidx, splitD['left'], attrs, stopUnity)
        trainTree(tree, ridx, splitD['right'], attrs, stopUnity)

if __name__ == '__main__':
    feature = 'default'
    fileIn = 'spam_data/spam_train.csv'
    fileOut = ''
    val = True
    scaling = True
    stopUnity = 0.9

    for i, arg in enumerate(sys.argv):
        if arg.startswith('-in'):
            fileIn = sys.argv[i+1]
        if arg.startswith('-out'):
            fileOut = sys.argv[i+1]
        if arg.startswith('-feat'):
            # -feat drop:wr,cf
            feature = sys.argv[i+1]
        if arg.startswith('-stop'):
            stopUnity = float(sys.argv[i+1])
        if arg.startswith('-noVal'):
            val = False
        if arg.startswith('-noScale'):
            scaling = False

    D = dataParser(fileIn, feature)
    if val:
        valScore = validate(D, stopUnity)
    else:
        valScore = 0

    tree = [0]
    attrs = list(D.columns)
    del(attrs[-1])
    print 'Training...'
    trainTree(tree, 0, D, attrs, stopUnity)
    print 'Done.'
    print tree

    model = {'tree':tree,
        'feature': feature,
        'stopUnity': stopUnity,
        'valScore': valScore, }
    if fileOut == '':
        fileOut = 'tval%06.4f_' % valScore
        fileOut += 'dim%02d_' % len(attrs)
        fileOut += feature
        fileOut += '.m'
    print 'Save as', fileOut, '?'
    raw_input()
    with open(fileOut, 'wb') as f:
        cPickle.dump(model, f)
        print  fileOut, 'saved.'
