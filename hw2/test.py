import pandas as pd
import numpy as np
import cPickle
import sys
from pprint import pprint
nameFile = 'spambase.names'
from train import readNames
from train import classify

def dataParser(filename, model):
    print 'Data preprocessing...'
    names = readNames(nameFile, train=False)
    df = pd.read_csv(filename, index_col=0, header=None, names=names)
    feature = model['feature']
    if feature.startswith('drop:'):
        dropItems = feature[5:].split(',')
        for item in dropItems:
            df = df.drop(item, axis=1, level=1, errors='ignore')
            df = df.drop(item, axis=1, level=2, errors='ignore')
    print '\tdf:', df.shape
    print 'Done.'
    
    return df

if __name__ == '__main__':
    fileModel = 'tree.m'
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
        print '\n', fileModel, 'loaded'
        print '\nFeature:', model['feature']
        print 'valScore:', model['valScore']
        print 'stopUnity:', model['stopUnity']
        print 'Tree nodes:', len(model['tree']), '\n'
    
    X = dataParser(fileTest, model)
    yGuess = []
    nameDic = {}
    for i, name in enumerate(X): 
        nameDic[name] = i

    for i, node in enumerate(model['tree']):
        if type(node) != int:
            model['tree'][i]['attr'] = nameDic[ node['attr'] ]
    raw_input()

    for x in X.values:
        yGuess.append ( classify( x,model['tree'] ) )

    if fileOut == '':
        fileOut = fileModel[:-2] + '.csv'
    print 'Save as', fileOut, '?'
    raw_input()

    with open(fileOut, 'w') as f:
        f.write('id,label\n')
        for i,y in enumerate(yGuess):
            f.write('%d,%d\n' % (i+1, y))
    print fileOut, 'Saved.'
