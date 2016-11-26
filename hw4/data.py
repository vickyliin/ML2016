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

def savetabel(table, path='table/'):
    size = len(table)/10
    for i in range(10):
        with open('%s%d.cpk'%(path,i), 'wb') as f:
            print('%d,\t%d : %d' % (i,size*i, size*(i+1)))
            cpk.dump(table[size*i:size*(i+1)], f)

def loadtable(path='table/'):
    table = []
    for i in range(10):
        with open('%s%d.cpk'%(path,i), 'rb') as f:
            print(i, end=', ', flush=True)
            table.append(cpk.load(f))
    table = np.array(table, dtype='bool').reshape(len(table[0])*10, -1)
    return table
    
def save(ans, filename='QAQ.csv'):
    start = time()
    with open(filename, 'w') as f:
        f.write('ID,Ans\n')
        for i in range(len(ans)):
            print('%d,%d'%(i,ans[i]), end='\r', flush=True)
            f.write('%d,%d\n'%(i,ans[i]))
    ela = time()-start
    print('\nWritten to file %s!\nTime: %.2f secs'%(filename, ela))
