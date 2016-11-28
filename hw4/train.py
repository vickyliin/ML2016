from sklearn.feature_extraction.text import *
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from time import time

from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.stats import variation
nb_class = 20
title_per_class = 1000

def FreqMatrixGen(T, min_df=2, max_df=1.0/nb_class, **kwargs):
    corpus = T['title']
    # analyzer
    stem = lambda terms_list:\
            [ LancasterStemmer().stem(term) for term in terms_list ]
    tokenize = CountVectorizer().build_analyzer()
    ana = lambda x: stem(tokenize(x))
    
    # vecterizer (tf)
    vectorizer = CountVectorizer(\
            min_df = min_df,
            max_df = max_df,
            analyzer = ana, 
            **kwargs)
    M = vectorizer.fit_transform(corpus)
    return M, vectorizer.get_feature_names()

def featExt(F, dimOut=100, normalize=True, **kwargs):
    # tfidf
    transformer = TfidfTransformer(smooth_idf=False)
    M = transformer.fit_transform(F)
    # LSA
    lsa = TruncatedSVD(n_components=dimOut, **kwargs)
    M = lsa.fit_transform(M) 
    # Normalize
    if normalize == True:
        M = normalize(M)
    return M

def cluster(M, n=nb_class, stop=0.1, **kwargs):
    cvar,i = 1,0
    while cvar > stop:
        print('\r#%d\t'%i, end='', flush='True')
        kmeans = KMeans(\
                n_clusters=n,
                n_init=1,
                **kwargs).fit(M)
        tags = kmeans.labels_
        count_tag = [ len(tags[tags==i]) for i in range(n)]
        cvar = variation(count_tag)
        print(cvar, end='', flush='True')
        i += 1
    return tags 
def bm25(idf, tf, fl, avgfl, B, K1):
    # idf - inverse document frequency
    # tf - term frequency in the current document
    # fl - field length in the current document
    # avgfl - average field length across documents in collection
    # B, K1 - free paramters

    return idf * ((tf * (K1 + 1)) / (tf + K1 * ((1 - B) + B * fl / avgfl)))


def check(C, tags):
    answer = np.zeros(len(C), dtype='bool')
    start = time()
    for i in C.index:
        answer[i] = tags[C['x_ID'][i]]==tags[C['y_ID'][i]]
        print('\r# %07d\t%d'%(i,answer[i]), end='', flush=True)
    ela = time()-start
    print('\nTime: %.2f secs'%ela)
    return answer

