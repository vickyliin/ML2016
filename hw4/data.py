from nltk.tokenize import word_tokenize
import pandas as pd
import _pickle as cpk
import numpy as np
from time import time

def loadT(path='data'):
    # load titles
    if path[-1] != '/':
        path += '/'
    with open(path+'title_StackOverflow.txt', 'r') as f:
        T={}
        T['title'] = [ x for x in f ]
        T['term'] = [ word_tokenize(x) for x in T['title'] ]
        T['tag'] = [0]*len(T['term'])
        T = pd.DataFrame(T)
    return T

def loadD(path='data'):
    # load docs
    if path[-1] != '/':
        path += '/'
    with open(path+'docs.txt', 'r') as f:
        sentences = [ word_tokenize(x) for x in f if x.strip() != '']
    return sentences

def loadC(path='data'):
    if path[-1] != '/':
        path += '/'
    df = pd.read_csv(path+'check_index.csv', index_col=0)
    return df

def saveT(tags, filename='T'):
    with open('%s.cpk'%filename, 'wb') as f:
        cpk.dump(tags, f)
        print('Save as %s.cpk!'%filename)

def loadT(filename='T'):
    with open('%s.cpk'%filename, 'rb') as f:
        tags = cpk.load(f)
        print('Model %s.cpk loaded!'%filename)
    return tags
    
def save(ans, filename='QAQ.csv'):
    start = time()
    #print('        M')
    with open(filename, 'w') as f:
        f.write('ID,Ans\n')
        for i in range(len(ans)):
            #print('\r# %d\t%d'%(i,ans[i]), end='', flush=True)
            f.write('%d,%d\n'%(i,ans[i]))
    ela = time()-start
    print('\nWritten to file %s!\nTime: %.2f secs'%(filename, ela))
