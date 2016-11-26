from gensim.models import Word2Vec
from whoosh import index
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, OrGroup
from whoosh.searching import Searcher, Results
import pandas as pd
import numpy as np
from time import time

class result():
    def __init__(self, r=0, searcher=0):
        if r==0:
            self.__dict__={ 
                'title':[],
                'num':[],
                'docnum':[],
                'score':[],
                'keyterm':[]}
        else:
            self.set(r, searcher)
    def set(self, r, searcher):
        if type(r) == Results:
            self.title  = [ x['title'] for x in r ]
            self.num    = [ x['num'] for x in r ]
            self.score  = [ x[0] for x in r.top_n ]
            self.docnum = [ x[1] for x in r.top_n ]
            key_score = [ searcher.key_terms([x], 'title') \
                    for x in self.docnum ]
            self.keyterm = [ list(zip(*x))[0] for x in key_score ]
    def df(self):
        return pd.DataFrame(self.__dict__)

def train_w2v(docs, output=None, **kwargs):
    model = Word2Vec(docs, **kwargs)
    if output != None:
        if output==1:
            output='vecmodel.cpk'
        model.save(output)
    return model

def indexGen(T, path='index'):
    schema = Schema( title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                     num=ID(stored=True) )
    ix = index.create_in(path, schema)
    with ix.writer() as w:
        for i in T.index:
            w.add_document(title=T['title'][i], num=str(i))
    return 0

def search(query, ix='index', limit=1000):
    ix = index.open_dir(ix)
    with ix.searcher() as s:
        parser = QueryParser('title', schema=ix.schema, group=OrGroup)
        q = parser.parse(query)
        rslt = result(s.search(q, limit=limit), s)
    return rslt

def build_table(T):
    ans = np.identity(20000, dtype='bool')
    for i in T.index:
        print(i, end=', ', flush=True)
        start=time()
        ans[i][ search(T['title'][i]).docnum ] = 1
        ela = time()-start
        print('%.2f sec' % ela, end='\t', flush=True)
    return ans

def check(C, table):
    answer = np.zeros(len(C), dtype='bool')
    start = time()
    for i in C.index:
        print(i, end=',', flush=True)
        answer[i] = table[ C['x_ID'][i] , C['y_ID'][i] ]
        print('%d'%answer[i], end='\t', flush=True)
    ela = time()-start
    print('\nTime: %.2f secs'%els)
    return answer

def make_sentVec(docs, wordmodel):
    return sentVecs
def train_sentModel(docVecs, output=None, **kwargs):
    return 0
